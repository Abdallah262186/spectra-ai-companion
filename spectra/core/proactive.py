"""
Proactive conversation initiator for Spectra.

Runs in a daemon background thread and, at random intervals, generates
context-aware openers that are injected into the terminal via the
ConversationManager.  Quiet hours and PC activity are respected.
"""

import logging
import random
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ProactiveSystem:
    """Background thread that prompts Spectra to start conversations proactively."""

    def __init__(
        self,
        config: Dict,
        engine,
        context_builder,
        database,
        inject_fn: Callable[[str], None],
        monitoring_manager=None,
    ) -> None:
        """Initialise the proactive system.

        Args:
            config: Full parsed config dictionary.
            engine: :class:`spectra.core.engine.AIEngine` instance.
            context_builder: :class:`spectra.memory.context.ContextBuilder` instance.
            database: :class:`spectra.memory.database.Database` instance.
            inject_fn: Callable that displays a proactive message in the REPL.
            monitoring_manager: Optional monitoring coordinator.
        """
        self.config = config
        self.engine = engine
        self.ctx = context_builder
        self.db = database
        self.inject_fn = inject_fn
        self.monitoring = monitoring_manager

        proactive_cfg = config.get("proactive", {})
        self.enabled: bool = proactive_cfg.get("enabled", True)
        self.min_interval: int = proactive_cfg.get("min_interval_minutes", 30) * 60
        self.max_interval: int = proactive_cfg.get("max_interval_minutes", 120) * 60

        companion_cfg = config.get("companion", {})
        self.quiet_start: int = companion_cfg.get("quiet_hours_start", 23)
        self.quiet_end: int = companion_cfg.get("quiet_hours_end", 8)
        self.companion_name: str = companion_cfg.get("name", "Spectra")

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background proactive thread."""
        if not self.enabled:
            logger.info("Proactive system disabled in config.")
            return
        self._thread = threading.Thread(target=self._loop, name="ProactiveThread", daemon=True)
        self._thread.start()
        logger.info("Proactive system started.")

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Proactive system stopped.")

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Sleep for a random interval then fire a proactive message."""
        while not self._stop_event.is_set():
            interval = random.randint(self.min_interval, self.max_interval)
            logger.debug("Proactive: next message in %d seconds.", interval)
            # Sleep in small chunks so we can respond to stop requests quickly
            for _ in range(interval):
                if self._stop_event.is_set():
                    return
                time.sleep(1)

            if self._stop_event.is_set():
                return

            if self._is_quiet_hours():
                logger.debug("Proactive: quiet hours — skipping.")
                continue

            self._fire()

    def _is_quiet_hours(self) -> bool:
        """Return True if the current time falls within quiet hours."""
        hour = datetime.now().hour
        if self.quiet_start > self.quiet_end:
            # e.g. 23 – 8 wraps midnight
            return hour >= self.quiet_start or hour < self.quiet_end
        return self.quiet_start <= hour < self.quiet_end

    def _fire(self) -> None:
        """Generate and inject one proactive message."""
        try:
            activities = self._get_activities()
            opener_prompt = self._build_opener_prompt(activities)
            system_prompt = self.ctx.build_system_prompt(current_activities=activities)

            message = self.engine.generate_response(
                user_message=opener_prompt,
                context=system_prompt,
                history=[],
                stream=False,
            )
            if message:
                self.inject_fn(message)
                self.db.save_activity("proactive", f"Proactive message sent: {message[:80]}")
        except Exception as exc:
            logger.error("Proactive fire error: %s", exc)

    def _get_activities(self) -> Optional[Dict]:
        """Collect current monitoring data."""
        if not self.monitoring:
            return None
        try:
            return self.monitoring.get_current_activities()
        except Exception as exc:
            logger.warning("Proactive monitoring error: %s", exc)
            return None

    def _build_opener_prompt(self, activities: Optional[Dict]) -> str:
        """Construct an internal prompt that nudges Spectra to open a conversation.

        Args:
            activities: Current PC activity dict or None.

        Returns:
            A short instruction for the model.
        """
        lines = [
            "You want to start a friendly, short proactive conversation with the user. "
            "Generate ONE opening sentence only — do not wait for their reply."
        ]
        if activities:
            if activities.get("spotify"):
                lines.append(
                    f"They are currently listening to: {activities['spotify']}. "
                    "You might comment on the music."
                )
            if activities.get("process"):
                lines.append(
                    f"They are currently: {activities['process']}. "
                    "You might check in on them."
                )

        hour = datetime.now().hour
        if 5 <= hour < 12:
            lines.append("It is morning — maybe wish them a good day.")
        elif 12 <= hour < 14:
            lines.append("It is around lunchtime — maybe suggest taking a break.")
        elif hour >= 21:
            lines.append("It is late evening — maybe suggest winding down.")

        return " ".join(lines)
