import logging
from typing import List, Dict, Any

from .base_agent import Agent
from ehr.store import EHRVectorStore


class EHRAgent(Agent):
    """
    Retrieval-augmented agent over an EHR vector store.

    Config keys (in configs/ehr_agent.yaml):
      - ehr_index_dir: path to a built index directory
      - embedding_model_name: sentence-transformers model name
      - retrieval_top_k: how many chunks to retrieve
      - context_max_chars: cap concatenated context size
      - agent_prompt: system prompt instructions
    """

    def __init__(self, settings_path, response_handler, message_bus=None):
        super().__init__(settings_path, response_handler, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)

        self.index_dir: str = self.agent_settings.get("ehr_index_dir", "ehr_index")
        self.embedding_model_name: str = self.agent_settings.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.top_k: int = int(self.agent_settings.get("retrieval_top_k", 5))
        self.context_max_chars: int = int(self.agent_settings.get("context_max_chars", 4000))

        # Lazy-load the vector store to avoid failing app startup if index is missing
        self.store = None
        self._logger.info(f"EHRAgent configured. Index dir: {self.index_dir}")

    def process_request(self, text: str, chat_history: List, visual_info: Dict[str, Any] | None = None):
        try:
            if self.store is None:
                try:
                    self._logger.info(f"Loading EHRVectorStore from {self.index_dir}")
                    self.store = EHRVectorStore.from_dir(self.index_dir, self.embedding_model_name)
                except Exception as e:
                    self._logger.error(f"Failed to load EHR index from {self.index_dir}: {e}")
                    return {
                        "name": "EHRAgent",
                        "response": (
                            f"EHR index not available at '{self.index_dir}'. "
                            "Build it with: python scripts/ehr_build_index.py"
                        ),
                    }

            retrieved = self.store.query(text, top_k=self.top_k)
            context_parts: List[str] = []
            total_chars = 0
            for r in retrieved:
                snippet = r.text.strip().replace("\n\n", "\n")
                header = f"[source: {r.metadata.get('source','unknown')} | chunk: {r.metadata.get('chunk_index','?')}]\n"
                block = header + snippet
                if total_chars + len(block) > self.context_max_chars:
                    break
                context_parts.append(block)
                total_chars += len(block)
            context = "\n\n".join(context_parts) if context_parts else "(no context retrieved)"

            # Build the user message combining question + retrieved context
            user_text = (
                "You are answering questions about a patient's EHR.\n"
                "Use ONLY the context below. If the answer is not present, say 'I don't know'.\n\n"
                f"Question: {text}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:"
            )

            prompt = self.generate_prompt(user_text, chat_history)
            response = self.stream_response(prompt=prompt, temperature=0.0)
            return {"name": "EHRAgent", "response": response}

        except Exception as e:
            self._logger.error(f"EHRAgent error: {e}", exc_info=True)
            return {"name": "EHRAgent", "response": f"Error: {str(e)}"}

