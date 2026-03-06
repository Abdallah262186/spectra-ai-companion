"""Training data preparation.

Queries the conversation database for turns not yet included in any prior
training run, converts them into Qwen2 ChatML instruction format, filters out
low-quality exchanges, and saves a JSONL file ready for the LoRA trainer.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MIN_CONTENT_LENGTH = 10  # characters; shorter messages are filtered out


class DataPrep:
    """Prepares fine-tuning data from the conversation history.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any) -> None:
        self.config = config
        self.db = db

    def prepare(self, output_path: str = "/tmp/spectra_train.jsonl") -> Tuple[int, str, int]:
        """Build and save the JSONL training file.

        Retrieves conversation rows not used in any previous training session,
        pairs them into (user, assistant) exchanges, filters low-quality pairs,
        and writes them as JSONL.

        Args:
            output_path: Destination file path for the JSONL data.

        Returns:
            Tuple of ``(number_of_samples, output_path, last_row_id)`` where
            ``number_of_samples`` is the count of written training pairs and
            ``last_row_id`` is the highest conversation id included (0 if none).
        """
        since_id = self.db.get_last_trained_id()
        rows = self.db.get_training_data(since_id=since_id)

        if not rows:
            logger.info("No new conversation data available for training.")
            return 0, output_path, 0

        pairs = self._pair_rows(rows)
        if not pairs:
            logger.info("No valid conversation pairs after filtering.")
            return 0, output_path, 0

        # Track the highest conversation id consumed by this run
        last_row_id = max(r["id"] for pair in pairs for r in pair)

        samples = self._to_chatml(pairs)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info("Saved %d training samples to %s", len(samples), output_path)
        return len(samples), output_path, last_row_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pair_rows(
        self, rows: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Pair consecutive (user, assistant) rows.

        Args:
            rows: List of row dicts ordered by id ascending.

        Returns:
            List of ``(user_row, assistant_row)`` tuples that pass quality
            filters.
        """
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        i = 0
        while i < len(rows) - 1:
            row_a = rows[i]
            row_b = rows[i + 1]
            if row_a.get("role") == "user" and row_b.get("role") == "assistant":
                if self._is_quality_pair(row_a["content"], row_b["content"]):
                    pairs.append((row_a, row_b))
                i += 2
            else:
                i += 1
        return pairs

    @staticmethod
    def _is_quality_pair(user_text: str, assistant_text: str) -> bool:
        """Return True if both messages are long enough to be useful.

        Args:
            user_text: User message content.
            assistant_text: Assistant message content.

        Returns:
            True if both pass the minimum length threshold.
        """
        # Skip proactive messages that were logged with a "[Proactive]" prefix
        if "[Proactive]" in assistant_text:
            return False
        return (
            len(user_text.strip()) >= _MIN_CONTENT_LENGTH
            and len(assistant_text.strip()) >= _MIN_CONTENT_LENGTH
        )

    @staticmethod
    def _to_chatml(
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Convert pairs to the ChatML messages format used by Qwen2.

        Args:
            pairs: List of ``(user_row, assistant_row)`` tuples.

        Returns:
            List of ``{"messages": [...]}`` dicts in JSONL-ready format.
        """
        samples = []
        for user_row, assistant_row in pairs:
            sample = {
                "messages": [
                    {"role": "user", "content": user_row["content"].strip()},
                    {"role": "assistant", "content": assistant_row["content"].strip()},
                ]
            }
            samples.append(sample)
        return samples
