"""
Data preparation for LoRA fine-tuning.

Converts raw conversation rows from the database into the Qwen2 ChatML
instruction format and writes a JSONL file ready for SFTTrainer.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum length thresholds (characters) to filter low-quality turns
MIN_USER_CHARS = 10
MIN_ASSISTANT_CHARS = 20


def group_turns_into_sessions(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """Group flat conversation rows by session_id.

    Args:
        rows: List of dicts with keys ``session_id``, ``role``, ``content``.

    Returns:
        Dict mapping session_id → ordered list of turn dicts.
    """
    sessions: Dict[str, List[Dict]] = {}
    for row in rows:
        sid = row.get("session_id", "unknown")
        sessions.setdefault(sid, []).append(row)
    return sessions


def session_to_chatml(turns: List[Dict]) -> Optional[str]:
    """Convert a list of turns into a single ChatML training example.

    Pairs of (user, assistant) turns are required.  The session is
    discarded if it contains no valid pairs.

    Args:
        turns: Ordered list of turn dicts (role + content).

    Returns:
        Formatted ChatML string ready for inclusion in JSONL, or None.
    """
    # Build (user_msg, assistant_msg) pairs from consecutive turns
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(turns) - 1:
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            user_text = turns[i]["content"].strip()
            asst_text = turns[i + 1]["content"].strip()
            if len(user_text) >= MIN_USER_CHARS and len(asst_text) >= MIN_ASSISTANT_CHARS:
                pairs.append((user_text, asst_text))
            i += 2
        else:
            i += 1

    if not pairs:
        return None

    lines: List[str] = []
    lines.append("<|im_start|>system\nYou are Spectra, a helpful personal AI companion.<|im_end|>")
    for user_text, asst_text in pairs:
        lines.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        lines.append(f"<|im_start|>assistant\n{asst_text}<|im_end|>")
    return "\n".join(lines)


def prepare_training_data(
    database,
    since_conversation_count: int = 0,
    output_path: Optional[str] = None,
) -> Tuple[str, int]:
    """Query the database and write a JSONL training file.

    Args:
        database: :class:`spectra.memory.database.Database` instance.
        since_conversation_count: Skip rows with ``id`` ≤ this value.
        output_path: Override for the output JSONL file path.

    Returns:
        Tuple of (output file path, number of training examples written).
    """
    rows = database.get_training_data(since_id=since_conversation_count)
    sessions = group_turns_into_sessions(rows)

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"training_data_{ts}.jsonl"

    examples_written = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for turns in sessions.values():
            text = session_to_chatml(turns)
            if text:
                fh.write(json.dumps({"text": text}) + "\n")
                examples_written += 1

    logger.info(
        "Prepared %d training examples from %d sessions → %s",
        examples_written,
        len(sessions),
        output_path,
    )
    return output_path, examples_written
