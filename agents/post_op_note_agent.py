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

import os
import json
import math
import logging
import re
from datetime import datetime
from .base_agent import Agent

class PostOpNoteAgent(Agent):
    def __init__(self, settings_path, response_handler=None, agent_key=None, message_bus=None):
        super().__init__(settings_path, response_handler, agent_key=agent_key, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)
        self.chunk_size = self.agent_settings.get("chunk_size", 20)

        self.schema_dict = {}
        if self.grammar:
            try:
                self.schema_dict = json.loads(self.grammar)
                self._logger.debug(f"Parsed grammar JSON schema: {self.schema_dict}")
            except json.JSONDecodeError as e:
                self._logger.error(f"Failed to parse 'grammar' as JSON: {e}")
        # Config defaults and mode flags
        defaults = self.agent_settings.get("defaults", {}) or {}
        self.default_procedure_type = defaults.get("procedure_type", "laparoscopic cholecystectomy")
        self.default_procedure_nature = defaults.get("procedure_nature", "unknown")
        # Merge personnel defaults: base -> global.yaml -> agent-specific
        try:
            global_personnel = (getattr(self, "global_settings", {}) or {}).get("personnel", {}) or {}
        except Exception:
            global_personnel = {}
        agent_personnel = (defaults.get("personnel", {}) or {})
        self.default_personnel = {
            "surgeon": "Not specified",
            "assistant": "Not specified",
            "anaesthetist": "Not specified",
        }
        # Global fallbacks, then agent overrides take precedence
        self.default_personnel.update(global_personnel)
        self.default_personnel.update(agent_personnel)
        mode = self.agent_settings.get("mode", {}) or {}
        self.llm_assist_findings = bool(mode.get("llm_assist_findings", True))

        # Smoothing and timeline options
        smoothing = self.agent_settings.get("smoothing", {}) or {}
        self.phase_min_dwell_seconds = int(smoothing.get("phase_min_dwell_seconds", 15))
        self.phase_min_consecutive = int(smoothing.get("phase_min_consecutive", 2))
        self.timeline_max_entries = int(smoothing.get("timeline_max_entries", 250))
        self.include_phase_summary = bool(smoothing.get("include_phase_summary", True))
        self.include_phase_details = bool(smoothing.get("include_phase_details", True))

    def process_request(self, input_data, chat_history, visual_info=None):
        return {
            "name": "PostOpNoteAgent",
            "response": "Invoke generate_post_op_note(procedure_folder) to produce and save the final note."
        }

    def generate_post_op_note(self, procedure_folder):
        try:
            self._logger.info(f"Starting post-op note generation for folder: {procedure_folder}")
            
            # Check if procedure folder exists
            if not os.path.isdir(procedure_folder):
                self._logger.error(f"Procedure folder does not exist: {procedure_folder}")
                return None
                
            annotation_json = os.path.join(procedure_folder, "annotation.json")
            notetaker_json = os.path.join(procedure_folder, "notetaker_notes.json")

            # Load annotations and notes
            self._logger.debug(f"Loading annotations from {annotation_json}")
            ann_list = self._load_json_array(annotation_json)
            if not ann_list:
                self._logger.warning("No annotation data found or unable to load annotations")
                
            self._logger.debug(f"Loading notes from {notetaker_json}")
            note_list = self._load_json_array(notetaker_json)
            if not note_list:
                self._logger.warning("No notetaker data found or unable to load notes")
                
            # Create default structure when data is missing (grammar-compliant)
            if not ann_list and not note_list:
                self._logger.warning("Both annotation and notetaker data are missing or empty - creating default structure")
                return {
                    "date_time": "Not specified",
                    "procedure_type": self.default_procedure_type,
                    "procedure_nature": self.default_procedure_nature,
                    "personnel": {
                        "surgeon": self.default_personnel.get("surgeon", "Not specified"),
                        "assistant": self.default_personnel.get("assistant", "Not specified"),
                        "anaesthetist": self.default_personnel.get("anaesthetist", "Not specified"),
                    },
                    "findings": "No findings recorded",
                    "complications": "None recorded",
                    "blood_loss_estimate": "Not specified",
                    "dvt_prophylaxis": "Not specified",
                    "antibiotic_prophylaxis": "Not specified",
                    "postoperative_instructions": "Not specified",
                    "timeline": []
                }
                
            # Deterministic extraction -> build note -> optional findings polish
            facts = self._extract_facts(ann_list, note_list)
            final_json = self._build_final_json_from_facts(facts)

            if self.llm_assist_findings:
                try:
                    refined = self._refine_findings_with_llm(facts, final_json.get("findings", ""))
                    if refined:
                        final_json["findings"] = refined
                except Exception as e:
                    self._logger.warning(f"LLM findings refinement failed; keeping deterministic findings. Error: {e}")

            post_op_file = os.path.join(procedure_folder, "post_op_note.json")
            self._save_post_op_note(final_json, post_op_file)

            return final_json
            
        except Exception as e:
            self._logger.error(f"Unexpected error in generate_post_op_note: {e}", exc_info=True)
            return None

    def _ask_for_json(self, prompt_text: str):
        messages = []
        if self.agent_prompt:
            messages.append({"role": "system", "content": self.agent_prompt})

        user_content = prompt_text.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
        messages.append({"role": "user", "content": user_content})

        self._logger.debug("Calling vLLM for JSON response.")
        try:
            # First attempt with standard parameters
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=self.ctx_length
            )
            content = result.choices[0].message.content
            # Strip any Python tag markers that might be in the response
            if content.startswith("<|python_tag|>"):
                content = content.replace("<|python_tag|>", "")
                
            # Check if we have a complete JSON response
            if self._is_truncated_json(content):
                self._logger.warning("Detected truncated JSON in first response, trying again with different parameters")
                # Try again with more structured approach
                structured_messages = [
                    {"role": "system", "content": f"{self.agent_prompt}\nIMPORTANT: Your response MUST be complete, valid JSON only. Do not truncate your response."},
                    {"role": "user", "content": f"Generate a post-operative note in JSON format. Keep it concise but complete.\n\n{user_content}"}
                ]
                
                # Second attempt with more explicit instructions and higher max_tokens
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=structured_messages,
                    temperature=0.2,  # Lower temperature for more deterministic output
                    max_tokens=self.ctx_length * 2  # Double the tokens to ensure completion
                )
                content = result.choices[0].message.content
            
            return content
        except Exception as e:
            self._logger.error(f"Error getting response from vLLM server: {e}", exc_info=True)
            raise
            
    def _is_truncated_json(self, text):
        """Check if JSON appears to be truncated"""
        # Count opening and closing braces
        open_braces = text.count('{')
        close_braces = text.count('}')
        
        # Check if we have complete pairs
        if open_braces != close_braces:
            self._logger.warning(f"Potential truncated JSON: {open_braces} opening braces vs {close_braces} closing braces")
            return True
            
        # Check for typical truncation patterns
        if text.rstrip().endswith(',') or text.rstrip().endswith(':') or text.rstrip().endswith('"'):
            self._logger.warning("Potential truncated JSON: ends with delimiter")
            return True
            
        try:
            # Try to parse it
            json.loads(text.strip())
            return False
        except json.JSONDecodeError as e:
            # If there's a specific truncation error
            if "Expecting" in str(e) or "Unterminated" in str(e):
                self._logger.warning(f"JSON parse error suggests truncation: {e}")
                return True
            
            # Otherwise it might be invalid for other reasons
            return False
            
    def _fix_truncated_json(self, text):
        """Attempt to fix truncated JSON by completing missing structure"""
        import re
        import datetime
        
        try:
            # Clean the text
            text = text.strip()
            
            # Extract what looks like a JSON object 
            if '{' in text:
                # Get everything from the first opening brace
                potential_json = text[text.find('{'):]
                
                # Count braces to determine what's missing
                open_braces = potential_json.count('{')
                close_braces = potential_json.count('}')
                
                # Add missing closing braces
                if open_braces > close_braces:
                    missing_braces = open_braces - close_braces
                    potential_json += '}' * missing_braces
                
                # Fix common truncation issues
                # Remove trailing commas before closing braces
                potential_json = re.sub(r',\s*}', '}', potential_json)
                
                # Fix unterminated strings by checking if we have an odd number of quotes
                if potential_json.count('"') % 2 != 0:
                    # Add a closing quote to the last string that's missing one
                    last_quote_pos = potential_json.rfind('"')
                    if last_quote_pos > 0:
                        potential_json = potential_json[:last_quote_pos+1] + '"' + potential_json[last_quote_pos+1:]
                
                try:
                    # Try to parse the fixed JSON
                    fixed_json = json.loads(potential_json)
                    self._logger.info("Successfully fixed truncated JSON")
                    
                    # Ensure it has the required structure
                    base_structure = {
                        "procedure_information": {
                            "procedure_type": "laparoscopic procedure",
                            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                            "duration": "Unknown",
                            "surgeon": "Not specified"
                        },
                        "findings": [],
                        "procedure_timeline": [],
                        "complications": []
                    }
                    
                    # Add any missing sections
                    for key, default in base_structure.items():
                        if key not in fixed_json:
                            fixed_json[key] = default
                        elif fixed_json[key] is None:
                            fixed_json[key] = default
                            
                    # Ensure procedure_information has all required fields
                    if isinstance(fixed_json.get("procedure_information"), dict):
                        for field, default in base_structure["procedure_information"].items():
                            if field not in fixed_json["procedure_information"]:
                                fixed_json["procedure_information"][field] = default
                    
                    return fixed_json
                    
                except Exception as e:
                    self._logger.warning(f"Failed to fix and parse truncated JSON: {e}")
            
            # Couldn't fix it, return None to trigger fallback
            return None
            
        except Exception as e:
            self._logger.error(f"Error in _fix_truncated_json: {e}", exc_info=True)
            return None

    # ----------------- Deterministic pipeline helpers -----------------
    def _parse_ts(self, ts: str):
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _format_duration(self, seconds):
        if seconds is None:
            return "Not specified"
        try:
            seconds = int(max(0, seconds))
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            return f"{h:02d}:{m:02d}:{s:02d}"
        except Exception:
            return "Not specified"

    def _extract_facts(self, ann_list, note_list):
        def ts_key(x):
            t = x.get("timestamp")
            dt = self._parse_ts(t) if isinstance(t, str) else None
            return dt or datetime.min

        anns = sorted([a for a in ann_list if isinstance(a, dict)], key=ts_key)
        notes = sorted([n for n in note_list if isinstance(n, dict)], key=ts_key)

        start_ts_str = None
        end_ts_str = None
        if anns:
            start_ts_str = anns[0].get("timestamp") or None
            end_ts_str = anns[-1].get("timestamp") or None
        elif notes:
            start_ts_str = notes[0].get("timestamp") or None
            end_ts_str = notes[-1].get("timestamp") or None

        duration_seconds = None
        for a in reversed(anns):
            if isinstance(a.get("elapsed_time_seconds"), (int, float)):
                duration_seconds = a["elapsed_time_seconds"]
                break
        if duration_seconds is None and start_ts_str and end_ts_str:
            d0 = self._parse_ts(start_ts_str)
            d1 = self._parse_ts(end_ts_str)
            if d0 and d1:
                duration_seconds = max(0, int((d1 - d0).total_seconds()))

        # Aggregate tools/anatomy (global)
        tools_set = set()
        anatomy_set = set()
        for a in anns:
            for t in a.get("tools", []) or []:
                if isinstance(t, str) and t != "none":
                    tools_set.add(t)
            for an in a.get("anatomy", []) or []:
                if isinstance(an, str) and an != "none":
                    anatomy_set.add(an)

        # Build a smoothed sequence of phases with dwell/consecutive thresholds
        smoothed_segments = []  # list of {phase, start_time}
        run_phase = None
        run_count = 0
        run_start_ts = None
        accepted_phase = None
        last_accept_dt = None

        for a in anns:
            phase = a.get("surgical_phase")
            ts = a.get("timestamp") or ""
            ts_dt = self._parse_ts(ts)
            if not isinstance(phase, str) or not ts_dt:
                continue

            if phase == run_phase:
                run_count += 1
            else:
                run_phase = phase
                run_count = 1
                run_start_ts = ts

            # Consider accepting a new phase when run_count and dwell satisfied
            if accepted_phase != run_phase and run_count >= self.phase_min_consecutive:
                dwell_ok = True
                if last_accept_dt is not None:
                    dwell_ok = (ts_dt - last_accept_dt).total_seconds() >= self.phase_min_dwell_seconds
                if dwell_ok:
                    smoothed_segments.append({"phase": run_phase, "start_time": run_start_ts})
                    accepted_phase = run_phase
                    last_accept_dt = ts_dt

        # Derive ordered phases and first seen times from smoothed segments
        phases_ordered = []
        phase_first_seen_time = {}
        for seg in smoothed_segments:
            ph = seg["phase"]
            st = seg["start_time"]
            if ph not in phase_first_seen_time:
                phase_first_seen_time[ph] = st
                phases_ordered.append(ph)

        # Compute durations for smoothed segments
        phase_durations = {}
        for i, seg in enumerate(smoothed_segments):
            ph = seg["phase"]
            st = seg["start_time"]
            st_dt = self._parse_ts(st)
            if not st_dt:
                continue
            if i + 1 < len(smoothed_segments):
                next_dt = self._parse_ts(smoothed_segments[i + 1]["start_time"]) or self._parse_ts(end_ts_str)
            else:
                next_dt = self._parse_ts(end_ts_str)
            if next_dt:
                phase_durations[ph] = phase_durations.get(ph, 0) + max(0, int((next_dt - st_dt).total_seconds()))

        # Build phase events from smoothed segments
        phase_events = [{"time": seg["start_time"], "event": f"Phase started: {seg['phase']}"} for seg in smoothed_segments]

        note_events = []
        complications_flags = []
        blood_loss_estimate = None
        antibiotic_prophylaxis = None
        dvt_prophylaxis = None
        abx_terms = [
            "cefazolin", "ancef", "cefoxitin", "ceftriaxone", "metronidazole", "zosyn",
            "piperacillin", "tazobactam", "augmentin", "amoxicillin", "ciprofloxacin",
            "levofloxacin", "ertapenem"
        ]
        dvt_terms = [
            "heparin", "enoxaparin", "lovenox", "lmwh", "compression boots", "boots",
            "sequential compression", "scd", "stockings"
        ]
        for n in notes:
            ts = n.get("timestamp") or ""
            txt = (n.get("text") or "").strip()
            if not txt:
                continue
            note_events.append({"time": ts, "event": f"Note: {txt}"})
            low = txt.lower()
            if any(k in low for k in ["bleed", "perforat", "converted", "complication", "leak", "injury", "spillage"]):
                complications_flags.append(txt)
            m = re.search(r"\b(?:ebl|blood\s*loss)\b\s*[:=-]?\s*(\d{1,5})\s*(ml|cc)?", low)
            if m and not blood_loss_estimate:
                val = m.group(1)
                unit = m.group(2) or "ml"
                blood_loss_estimate = f"{val} {unit}"
            if any(term in low for term in abx_terms) and not antibiotic_prophylaxis:
                antibiotic_prophylaxis = txt
            if any(term in low for term in dvt_terms) and not dvt_prophylaxis:
                dvt_prophylaxis = txt

        timeline = sorted(phase_events + note_events, key=lambda e: self._parse_ts(e.get("time")) or datetime.min)
        # Cap timeline length if configured
        if self.timeline_max_entries and len(timeline) > self.timeline_max_entries:
            omitted = len(timeline) - self.timeline_max_entries
            keep_head = min(50, self.timeline_max_entries // 4)
            keep_tail = self.timeline_max_entries - keep_head
            ellipsis_event = {
                "time": start_ts_str or "Not specified",
                "event": f"… {omitted} events omitted …",
            }
            timeline = timeline[:keep_head] + [ellipsis_event] + timeline[-keep_tail:]
        # Keep a compact copy of annotations for optional phase details
        anns_compact = []
        for a in anns:
            anns_compact.append({
                "timestamp": a.get("timestamp"),
                "tools": a.get("tools", []),
                "anatomy": a.get("anatomy", []),
            })

        facts = {
            "start_time": start_ts_str,
            "end_time": end_ts_str,
            "duration_seconds": duration_seconds,
            "phases_ordered": phases_ordered,
            "phase_first_seen_time": phase_first_seen_time,
            "phase_durations": phase_durations,
            "tools": sorted(tools_set),
            "anatomy": sorted(anatomy_set),
            "timeline": timeline,
            "annotations": anns_compact,
            "complications_flags": complications_flags,
            "blood_loss_estimate": blood_loss_estimate,
            "antibiotic_prophylaxis": antibiotic_prophylaxis,
            "dvt_prophylaxis": dvt_prophylaxis,
        }
        self._logger.debug(f"Extracted facts: {facts}")
        return facts

    def _build_final_json_from_facts(self, facts: dict) -> dict:
        phases = facts.get("phases_ordered", [])
        tools = facts.get("tools", [])
        anatomy = facts.get("anatomy", [])
        dur_str = self._format_duration(facts.get("duration_seconds"))
        findings_parts = []
        if phases:
            findings_parts.append(f"Phases observed: {', '.join(phases)}.")
        if tools:
            findings_parts.append(f"Tools seen: {', '.join(tools)}.")
        if anatomy:
            findings_parts.append(f"Anatomy involved: {', '.join(anatomy)}.")
        if dur_str and dur_str != "Not specified":
            findings_parts.append(f"Approximate procedure duration: {dur_str}.")
        if not findings_parts:
            findings_parts.append("Findings: Not specified.")

        complications = "None recorded"
        if facts.get("complications_flags"):
            complications = "; ".join(facts["complications_flags"])[:500]

        post_op = {
            "date_time": facts.get("start_time") or "Not specified",
            "procedure_type": self.default_procedure_type,
            "procedure_nature": self.default_procedure_nature,
            "personnel": {
                "surgeon": self.default_personnel.get("surgeon", "Not specified"),
                "assistant": self.default_personnel.get("assistant", "Not specified"),
                "anaesthetist": self.default_personnel.get("anaesthetist", "Not specified"),
            },
            "findings": " ".join(findings_parts),
            "complications": complications,
            "blood_loss_estimate": facts.get("blood_loss_estimate") or "Not specified",
            "dvt_prophylaxis": facts.get("dvt_prophylaxis") or "Not specified",
            "antibiotic_prophylaxis": facts.get("antibiotic_prophylaxis") or "Not specified",
            "postoperative_instructions": "Not specified",
            "timeline": [
                {"time": (e.get("time") or "Not specified"), "event": (e.get("event") or "")[:500]}
                for e in facts.get("timeline", [])
            ],
        }
        # Optional extras: per‑phase summary and details
        if self.include_phase_summary:
            phase_summary = []
            for ph in facts.get("phases_ordered", []):
                st = facts.get("phase_first_seen_time", {}).get(ph)
                dur = facts.get("phase_durations", {}).get(ph)
                phase_summary.append({
                    "phase": ph,
                    "start_time": st or "Not specified",
                    "duration": self._format_duration(dur),
                    "duration_seconds": int(dur) if isinstance(dur, (int, float)) else None,
                })
            post_op["phase_summary"] = phase_summary

        if self.include_phase_details:
            # Build details by splitting annotations into segments based on smoothed start times
            details = []
            # Reconstruct smoothed segments as in facts
            segs = []
            for ph in facts.get("phases_ordered", []):
                st = facts.get("phase_first_seen_time", {}).get(ph)
                if st:
                    segs.append({"phase": ph, "start_time": st})
            segs = sorted(segs, key=lambda s: self._parse_ts(s["start_time"]) or datetime.min)
            for i, seg in enumerate(segs):
                ph = seg["phase"]
                st_dt = self._parse_ts(seg["start_time"]) or datetime.min
                en_dt = self._parse_ts(segs[i+1]["start_time"]) if i+1 < len(segs) else self._parse_ts(facts.get("end_time"))
                seg_tools = set()
                seg_anat = set()
                for a in facts.get("annotations", []):
                    at = self._parse_ts(a.get("timestamp"))
                    if not at:
                        continue
                    if at >= st_dt and (en_dt is None or at < en_dt):
                        for t in a.get("tools", []) or []:
                            if isinstance(t, str) and t != "none":
                                seg_tools.add(t)
                        for an in a.get("anatomy", []) or []:
                            if isinstance(an, str) and an != "none":
                                seg_anat.add(an)
                details.append({"phase": ph, "tools": sorted(seg_tools), "anatomy": sorted(seg_anat)})
            post_op["phase_details"] = details
        return post_op

    def _refine_findings_with_llm(self, facts: dict, draft_findings: str) -> str:
        lines = []
        lines.append(f"Phases: {', '.join(facts.get('phases_ordered', [])) or 'None'}")
        lines.append(f"Tools: {', '.join(facts.get('tools', [])) or 'None'}")
        lines.append(f"Anatomy: {', '.join(facts.get('anatomy', [])) or 'None'}")
        dur = self._format_duration(facts.get("duration_seconds"))
        lines.append(f"Duration: {dur}")
        fact_sheet = "\n".join(lines)

        system = (
            "You rewrite the 'findings' sentence for a post‑operative note. "
            "You MUST ONLY rephrase the facts provided. Do not add any new facts, numbers, or names. "
            "If a field is 'Not specified' or 'None', do not invent it. Keep it concise and clinical."
        )
        user = (
            f"Facts (verbatim):\n{fact_sheet}\n\n"
            f"Draft findings to polish:\n{draft_findings}\n\n"
            "Rephrase into 1–3 concise sentences using only the facts."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=self.ctx_length,
        )
        return (result.choices[0].message.content or "").strip()

    def _chunk_summarize_annotation(self, ann_list):
        if not ann_list:
            return "No annotation data found."
        
        lines = []
        
        # Track all unique tools, phases, and anatomy for comprehensive summary
        all_tools = set()
        all_phases = set()
        all_anatomy = set()
        
        for ann in ann_list:
            ts = ann.get("timestamp", "???")
            phase = ann.get("surgical_phase", "?")
            desc = ann.get("description", "?")
            
            # Extract and track tools, anatomy
            tools = ann.get("tools", [])
            anatomy = ann.get("anatomy", [])
            
            # Update our tracking sets
            if phase and phase != "?":
                all_phases.add(phase)
            if tools:
                all_tools.update(tools)
            if anatomy:
                all_anatomy.update(anatomy)
            
            # Create a more detailed line including tools and anatomy when available
            details = []
            if tools:
                details.append(f"Tools=[{', '.join(tools)}]")
            if anatomy:
                details.append(f"Anatomy=[{', '.join(anatomy)}]")
                
            if details:
                lines.append(f"[{ts}] Phase={phase}, {desc} {' '.join(details)}")
            else:
                lines.append(f"[{ts}] Phase={phase}, {desc}")
        
        # Add a summary line at the beginning to highlight all tools, phases, and anatomy
        summary_lines = []
        if all_phases:
            summary_lines.append(f"ALL PHASES: {', '.join(all_phases)}")
        if all_tools:
            summary_lines.append(f"ALL TOOLS: {', '.join(all_tools)}")
        if all_anatomy:
            summary_lines.append(f"ALL ANATOMY: {', '.join(all_anatomy)}")
            
        # Combine the summary header with the detailed lines
        if summary_lines:
            lines = summary_lines + ["---"] + lines
            
        return self._multi_step_chunk_summarize(lines, label="Annotation data")

    def _chunk_summarize_notetaker(self, note_list):
        if not note_list:
            return "No notetaker data found."
            
        # Log the actual count of notes for debugging
        self._logger.info(f"Processing {len(note_list)} notetaker notes")

        # Filter out empty or placeholder notes
        valid_notes = []
        for note in note_list:
            text = note.get("text", "").strip()
            title = note.get("title", "").strip()
            
            # Skip notes with empty or placeholder content
            if not text or text.lower() in ["take a note", "no text", "empty"]:
                self._logger.debug(f"Skipping empty/placeholder note: {note}")
                continue
                
            valid_notes.append(note)
            
        self._logger.info(f"Found {len(valid_notes)} valid notes after filtering")
        
        if not valid_notes:
            return "No substantive notetaker data found (0 valid notes)."

        lines = []
        # Add a note count header
        lines.append(f"TOTAL NOTES: {len(valid_notes)}")
        lines.append("---")
        
        for note in valid_notes:
            ts = note.get("timestamp", "???")
            txt = note.get("text", "(no text)")
            title = note.get("title", "")
            
            # Include the title if available
            if title:
                lines.append(f"[{ts}] TITLE: {title} | CONTENT: {txt}")
            else:
                lines.append(f"[{ts}] {txt}")

        return self._multi_step_chunk_summarize(lines, label="Notetaker data")

    def _multi_step_chunk_summarize(self, lines, label="Data"):
        # If no lines to summarize, return a default message
        if not lines:
            return f"No {label.lower()} available to summarize."
            
        if len(lines) <= self.chunk_size:
            block = "\n".join(lines)
            return self._ask_for_summary(block, label)
        else:
            try:
                chunk_summaries = []
                total = len(lines)
                n_chunks = math.ceil(total / self.chunk_size)
                idx = 0
                for i in range(n_chunks):
                    chunk = lines[idx:idx+self.chunk_size]
                    idx += self.chunk_size
                    chunk_text = "\n".join(chunk)
                    sub_summary = self._ask_for_summary(chunk_text, f"{label} chunk {i+1}/{n_chunks}")
                    if sub_summary: # Only add non-empty summaries
                        chunk_summaries.append(sub_summary)
                
                # If all chunk summaries failed, return a default message
                if not chunk_summaries:
                    return f"Unable to generate summary for {label.lower()}."
                    
                final_block = "\n\n".join(chunk_summaries)
                final_summary = self._ask_for_summary(final_block, f"{label} final summary")
                
                # If final summary is empty, use the first chunk summary
                if not final_summary and chunk_summaries:
                    final_summary = chunk_summaries[0]
                    
                return final_summary or f"Unable to generate final summary for {label.lower()}."
                
            except Exception as e:
                self._logger.error(f"Error in multi-step chunk summarization: {e}", exc_info=True)
                return f"Error summarizing {label.lower()}: {str(e)}"

    def _ask_for_summary(self, text_block, label="Data"):
        messages = []
        if self.agent_prompt:
            messages.append({"role": "system", "content": self.agent_prompt})

        user_prompt = (
            f"You are summarizing {label}.\n"
            f"Here is the data:\n\n{text_block}\n\n"
            "Please produce a concise summary.\n"
        )
        messages.append({"role": "user", "content": user_prompt})

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5,
                max_tokens=self.ctx_length
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            self._logger.error(f"Error summarizing {label} with vLLM: {e}")
            return ""

    def _load_json_array(self, filepath):
        if not os.path.isfile(filepath):
            self._logger.warning(f"File not found: {filepath}")
            return []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                self._logger.debug(f"Loaded data from {filepath}: {data[:500] if len(str(data)) > 500 else data}")
                
                if not isinstance(data, list):
                    self._logger.warning(f"{filepath} is not a JSON list.")
                    return []
                
                # Check if we have valid content or just empty placeholders
                if not data:
                    self._logger.warning(f"{filepath} is an empty list.")
                    return []
                
                # Log the actual count of items
                self._logger.info(f"Loaded {len(data)} items from {filepath}")
                
                # For notetaker notes, we'll filter, not exclude completely
                if "notetaker_notes.json" in filepath:
                    # Check if we have at least one valid note
                    has_valid_note = any(
                        isinstance(item, dict) and 
                        item.get("text", "").strip() and 
                        item.get("text", "").lower().strip() not in ["", "take a note"]
                        for item in data
                    )
                    
                    if not has_valid_note:
                        self._logger.warning(f"{filepath} contains only empty or placeholder notes.")
                        return []
                    
                    return data
                
                # For annotation files, keep any non-empty list
                return data
        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid JSON in {filepath}: {e}", exc_info=True)
            return []
        except Exception as e:
            self._logger.error(f"Error reading {filepath}: {e}", exc_info=True)
            return []

    def _save_post_op_note(self, note_json, filepath):
        try:
            with open(filepath, "w") as f:
                json.dump(note_json, f, indent=2)
            self._logger.info(f"Post-op note saved to: {filepath}")
        except Exception as e:
            self._logger.error(f"Error writing post-op note to {filepath}: {e}", exc_info=True)
