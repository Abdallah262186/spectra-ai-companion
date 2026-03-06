"""Proactive conversation initiator.

Runs in a daemon thread and wakes up at randomised intervals (30–120 min by
default) to compose a context-aware opener and print it to the terminal.
Respects configured quiet hours so it never interrupts at night.
"""

import logging
import random
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProactiveSystem(threading.Thread):
    """Background thread that initiates proactive conversations.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
        engine: Loaded :class:`~spectra.core.engine.AIEngine` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any, engine: Any) -> None:
        super().__init__(daemon=True, name="ProactiveSystem")
        self.config = config
        self.db = db
        self.engine = engine
        self.companion_name: str = config.get("companion", {}).get("name", "Spectra")
        pro_cfg = config.get("proactive", {})
        self.min_interval: int = pro_cfg.get("min_interval_minutes", 30) * 60
        self.max_interval: int = pro_cfg.get("max_interval_minutes", 120) * 60
        cmp_cfg = config.get("companion", {})
        self.quiet_start: int = cmp_cfg.get("quiet_hours_start", 23)
        self.quiet_end: int = cmp_cfg.get("quiet_hours_end", 8)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the thread to stop and wake it up from sleep."""
        self._stop_event.set()

    def run(self) -> None:
        """Thread entry point – loop indefinitely, sleeping between messages."""
        logger.info("Proactive system started.")
        while not self._stop_event.is_set():
            interval = random.randint(self.min_interval, self.max_interval)
            logger.debug("Proactive system sleeping for %d s.", interval)
            self._stop_event.wait(interval)

            if self._stop_event.is_set():
                break

            if self._is_quiet_hours():
                logger.debug("Quiet hours – skipping proactive message.")
                continue

            self._send_proactive_message()

        logger.info("Proactive system stopped.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_quiet_hours(self) -> bool:
        """Return True if the current time falls within configured quiet hours."""
        now_hour = datetime.now().hour
        if self.quiet_start > self.quiet_end:
            # E.g. 23–08 spans midnight
            return now_hour >= self.quiet_start or now_hour < self.quiet_end
        return self.quiet_start <= now_hour < self.quiet_end

    def _build_opener_prompt(self) -> str:
        """Build a prompt asking the model to generate a proactive opener.

        Returns:
            A short instruction string for the model.
        """
        context_parts = []

        # Current activity context
        try:
            activities = self.db.get_recent_activities(limit=3)
            if activities:
                activity_lines = []
                for act in activities:
                    activity_lines.append(
                        f"  - [{act.get('activity_type', '?')}] {act.get('details', '')}"
                    )
                context_parts.append("Recent user activities:\n" + "\n".join(activity_lines))
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Could not fetch activities: %s", exc)

        time_str = datetime.now().strftime("%H:%M")
        context_parts.append(f"Current time: {time_str}")

        context_block = "\n".join(context_parts)
        return (
            f"You are {self.companion_name}, a friendly AI companion. "
            "Generate ONE short (1–2 sentence) proactive conversation opener based on the "
            "user's recent activities. Be natural and curious. Do not ask multiple questions.\n\n"
            f"{context_block}\n\nOpener:"
        )

    def _send_proactive_message(self) -> None:
        """Generate and print a proactive message to the terminal."""
        try:
            prompt = self._build_opener_prompt()
            messages = [{"role": "user", "content": prompt}]
            response = self.engine.generate_response(messages, stream=False)
            if response:
                # Use ANSI yellow directly to avoid requiring colorama import here
                print(f"\n\033[33m[{self.companion_name}]\033[0m {response}\n", flush=True)
                self.db.save_message(
                    role="assistant",
                    content=f"[Proactive] {response}",
                    session_id="proactive",
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Proactive message failed: %s", exc)
