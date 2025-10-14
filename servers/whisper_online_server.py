#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
import os
import logging
import numpy as np
import socket
import io
import subprocess
import time
import math
import signal
import errno

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile
import librosa

# If you plan to use OpenAI API or faster-whisper,
# be sure to install them, e.g.:
#   pip install openai
#   pip install faster-whisper
# Otherwise, remove/comment out the relevant code below.

logger = logging.getLogger(__name__)


def _terminate_processes_using_port(port: int, host: str = "0.0.0.0") -> bool:
    """Attempt to terminate processes already bound to the given TCP port."""
    try:
        # lsof returns PIDs listening on the port; `-nP` for speed and no DNS lookups
        output = subprocess.check_output(["lsof", "-nP", "-t", f"-iTCP:{port}"])
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        logger.error("lsof not available; cannot free port %s", port)
        return False

    killed_any = False
    current_pid = os.getpid()

    for line in output.decode().strip().splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue

        if pid == current_pid:
            continue

        try:
            os.kill(pid, signal.SIGTERM)
            killed_any = True
            logger.warning("Sent SIGTERM to PID %s using port %s", pid, port)
        except ProcessLookupError:
            continue
        except PermissionError as e:
            logger.error("Insufficient permissions to terminate PID %s: %s", pid, e)

    if killed_any:
        # give processes a moment to exit cleanly
        time.sleep(0.5)

    return killed_any

###############################################################################
# BEGIN: Inline "whisper_online.py" essentials
###############################################################################

from functools import lru_cache

@lru_cache(10**6)
def load_audio(fname):
    """
    Load entire file from disk, convert to 16kHz float32 numpy array using librosa.
    """
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    """
    Load a slice [beg, end) from the 16kHz audio,
    in seconds.
    """
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

class ASRBase:
    """
    Abstract base class that covers the minimal interface:
      - load_model() in child
      - transcribe(audio, init_prompt="")
      - optional use_vad() and set_translate_task()
    """
    sep = " "  # define how words are joined in final text output

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir=None, model_dir=None):
        raise NotImplementedError("Child class must implement load_model()")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("Child class must implement transcribe()")

    def use_vad(self):
        pass

    def set_translate_task(self):
        pass


class WhisperTimestampedASR(ASRBase):
    """
    Uses the 'whisper_timestamped' library as the backend.
    Slower than faster-whisper, but simpler setup for GPU.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped

        if model_dir is not None:
            logger.debug("WhisperTimestampedASR: ignoring model_dir, not implemented")

        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs
        )
        return result

    def ts_words(self, r):
        """
        Convert a whisper_timestamped result to a list of word-level timestamps:
         [(start_sec, end_sec, word), ...]
        """
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                o.append((w["start"], w["end"], w["text"]))
        return o

    def segments_end_ts(self, res):
        """
        Return list of segment end times for chunk-based trimming logic.
        """
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """
    Uses 'faster-whisper' library as the backend. Very fast but must be installed separately.
    """

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            logger.debug(f"FasterWhisperASR: loading model from {model_dir}. modelsize/cache_dir not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("FasterWhisperASR: must specify modelsize or model_dir.")
        # if no GPU use below
        # model = WhisperModel(model_size_or_path, device="cpu", compute_type="int8", download_root=cache_dir)
        model = WhisperModel(model_size_or_path, device="cuda", compute_type="int8", download_root=cache_dir)

        return model

    def transcribe(self, audio, init_prompt=""):
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,                # a decent trade-off: better quality than greedy, but still pretty fast
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs
        )
        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            # skip if no_speech_prob is too high
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                o.append((word.start, word.end, word.word))
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """
    Uses OpenAI's Whisper API (whisper-1) for transcription.
    Requires 'openai' Python package and valid API key.
    """

    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        super().__init__(lan=lan, logfile=logfile)  # we have to init self.transcribe_kargs
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.task = "transcribe"
        self.transcribed_seconds = 0

    def load_model(self, *args, **kwargs):
        import openai
        # We'll just store a reference to the openai module;
        # actual calls happen in transcribe().
        self.client = openai

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        import soundfile as sf
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        # Track total length for cost estimation
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"]
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt

        if self.task == "translate":
            proc = self.client.Audio.translate
        else:
            proc = self.client.Audio.transcribe

        transcript = proc(**params)
        logger.debug(f"OpenAI API processed a total of {self.transcribed_seconds} seconds so far.")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"


DEFAULT_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh"
# Override if WHISPER_LANG_CODES environment variable is set
WHISPER_LANG_CODES = os.getenv("WHISPER_LANG_CODES", DEFAULT_LANG_CODES).split(",")

def create_tokenizer(lan):
    """
    Returns an object that has a .split() function for sentence segmentation.
    The server does not currently use it, but we keep it for code completeness.
    """
    # For demonstration, just a simple fallback
    class SimpleTokenizer:
        def split(self, text):
            # naive sentence splitting on periods, question marks, exclamation marks
            import re
            chunks = re.split(r"([.?!])", text)
            sentences = []
            tmp = ""
            for part in chunks:
                tmp += part
                if part in [".", "?", "!"]:
                    sentences.append(tmp.strip())
                    tmp = ""
            if tmp.strip():
                sentences.append(tmp.strip())
            return sentences
    return SimpleTokenizer()

def add_shared_args(parser):
    """
    Add the Whisper-related CLI arguments for the server or other scripts.
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0,
                        help='Minimum audio chunk size in seconds (unused by server).')
    parser.add_argument('--model', type=str, default='large-v2',
                        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(","),
                        help="Name/size of the Whisper model.")
    parser.add_argument('--model_cache_dir', type=str, default=None,
                        help="Optional local dir for caching downloaded models.")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Dir with already-downloaded Whisper model, overrides --model.")
    parser.add_argument('--lan', '--language', type=str, default='auto',
                        help="Whisper language code or 'auto'.")
    parser.add_argument('--task', type=str, default='transcribe',
                        choices=["transcribe","translate"],
                        help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper",
                        choices=["faster-whisper", "whisper_timestamped", "openai-api"],
                        help="Which backend to use.")
    parser.add_argument('--vac', action="store_true", default=False,
                        help='(Unused by server) Voice activity controller, not used here.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04,
                        help='(Unused by server) VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False,
                        help='Use built-in VAD if supported by backend.')
    parser.add_argument('--buffer_trimming', type=str, default="segment",
                        choices=["sentence", "segment"],
                        help='(Unused by server) Trimming strategy if streaming.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15,
                        help='(Unused by server) Trim length threshold in seconds.')
    parser.add_argument("-l", "--log-level", dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the log level", default='DEBUG')

def asr_factory(args, logfile=sys.stderr):
    """
    Creates an ASR object based on the specified backend and arguments.
    The server only needs the 'asr' part, not the streaming logic.
    """
    backend = args.backend

    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        else:
            asr_cls = WhisperTimestampedASR

        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = asr_cls(modelsize=size,
                      lan=args.lan,
                      cache_dir=args.model_cache_dir,
                      model_dir=args.model_dir)
        e = time.time()
        logger.info(f"Done loading. It took {round(e - t, 2)} seconds.")

    # if user requested VAD from the command line, let the ASR know
    if getattr(args, 'vad', False):
        logger.info("Enabling VAD in the chosen backend.")
        asr.use_vad()

    # if user wants translation, set it
    if args.task == "translate":
        asr.set_translate_task()

    # We do not create an "online" object for partial streaming here—just the ASR.
    # Return a dummy placeholder for `_unused_online`.
    return asr, None

def set_logging(args, logger, other=""):
    """
    Configure basic logging. 
    """
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    # If needed, you can also set other modules' logging here:
    # logging.getLogger("whisper_online"+other).setLevel(args.log_level)

###############################################################################
# END: Inline "whisper_online.py" essentials
###############################################################################


# =============================================================================
# =           Below is the original server code, now self-contained           =
# =============================================================================

parser = argparse.ArgumentParser()

# Server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43001)
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
                    help="WAV file to warm up Whisper so first chunk is faster.")

# Options from the inlined whisper_online code
add_shared_args(parser)
args = parser.parse_args()

# Logging config
set_logging(args, logger, other="")

SAMPLING_RATE = 16000

# Prepare the Whisper model
asr, _unused_online = asr_factory(args)
min_chunk = args.min_chunk_size

# Optional warm-up
if args.warmup_file and os.path.isfile(args.warmup_file):
    # Just load 1 second of the warmup file to force model initialization
    a = load_audio_chunk(args.warmup_file, 0, 1)
    asr.transcribe(a)
    logger.info("Whisper warmed up with warmup_file.")
else:
    logger.warning("No warmup file or not found. First transcription might be slower.")

def decode_webm_to_raw_pcm(webm_data):
    """
    Use ffmpeg to decode WebM/Opus -> raw PCM s16le @16kHz mono.
    Returns raw bytes (16-bit little-endian).
    """
    if not webm_data:
        logger.error("No webm_data was provided to decode_webm_to_raw_pcm()!")
        return b""

    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",        # read from stdin
            "-ac", "1",            # mono
            "-ar", "16000",        # 16k sample rate
            "-f", "s16le",         # raw PCM
            "pipe:1"               # write to stdout
        ],
        input=webm_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if process.returncode != 0:
        logger.error(f"ffmpeg decode error:\n{process.stderr.decode('utf-8','replace')}")
        return b""
    return process.stdout

def transcribe_full_webm(webm_data):
    """
    Takes raw WebM/Opus data, decodes via ffmpeg, then transcribes with Whisper.
    Returns recognized text as a single final string.
    """
    raw_pcm = decode_webm_to_raw_pcm(webm_data)
    if not raw_pcm:
        return ""

    # Convert raw PCM int16 -> float32
    sf = soundfile.SoundFile(
        io.BytesIO(raw_pcm),
        channels=1,
        samplerate=SAMPLING_RATE,
        format="RAW",
        subtype="PCM_16",
        endian="LITTLE"
    )
    # librosa.load() to produce a float32 numpy array
    audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)

    # Single final decode
    result = asr.transcribe(audio)

    # Convert `result` to final text string depending on the backend
    if args.backend == "faster-whisper":
        # result is a list of segments
        text = "".join(seg.text for seg in result)
    elif args.backend == "whisper_timestamped":
        # result is a dict with "segments"
        text = "".join(s["text"] for s in result["segments"])
    else:
        # For openai-api or fallback
        if hasattr(result, 'text'):
            # e.g. openai API
            text = result.text
        elif isinstance(result, dict) and "text" in result:
            # openai or other
            text = result["text"]
        else:
            # Possibly a plain string or unknown structure
            text = str(result)
    return text.strip()

class Connection:
    """
    Thin wrapper around a socket to receive all data, then send a response line.
    """
    PACKET_SIZE = 65536

    def __init__(self, conn):
        self.conn = conn
        self.conn.setblocking(True)

    def receive_all(self):
        """
        Receive all incoming bytes until client closes or resets connection.
        """
        full_data = bytearray()
        while True:
            try:
                chunk = self.conn.recv(self.PACKET_SIZE)
                if not chunk:
                    break
                full_data.extend(chunk)
            except ConnectionResetError:
                break
        return bytes(full_data)

    def send_line(self, line):
        """
        Send one line of UTF-8 text back.
        """
        try:
            line_bytes = (line + "\n").encode('utf-8', 'replace')
            self.conn.sendall(line_bytes)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Client connection lost while sending data: {e}")
            # Don't raise - let the connection close gracefully

def serve_one_connection(conn):
    """
    Handle a single client connection:
      - read entire WebM/Opus data
      - decode + transcribe
      - send recognized text
    """
    try:
        c = Connection(conn)
        webm_data = c.receive_all()
        logger.debug(f"Received {len(webm_data)} bytes from client. Decoding + Transcribing...")

        # Check if we received valid data
        if len(webm_data) == 0:
            logger.warning("Received empty data from client")
            c.send_line("0 0 ")
            return

        text = transcribe_full_webm(webm_data)

        logger.debug(f"Final recognized text: {text}")
        if text:
            # Return one line, e.g. "0 0 recognized_text"
            c.send_line(f"0 0 {text}")
        else:
            c.send_line("0 0 ")

    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        logger.warning(f"Connection error while serving client: {e}")
    except Exception as e:
        logger.error(f"Unexpected error serving client: {e}")
    finally:
        try:
            conn.close()
            logger.info("Connection to client closed")
        except:
            pass

# Main server loop
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Enable socket reuse to avoid "Address already in use" errors
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    for attempt in range(2):
        try:
            s.bind((args.host, args.port))
            break
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or attempt == 1:
                raise
            logger.warning(
                "Port %s already in use; attempting to terminate existing listener", args.port
            )
            if not _terminate_processes_using_port(args.port, args.host):
                logger.error("Could not free port %s; aborting.", args.port)
                raise
            # Loop will retry bind

    s.listen(1)
    logger.info(f"Listening on {(args.host, args.port)}")
    while True:
        conn, addr = s.accept()
        logger.info(f"Connected to client on {addr}")
        serve_one_connection(conn)

logger.info("Terminating whisper_online_server.py")
