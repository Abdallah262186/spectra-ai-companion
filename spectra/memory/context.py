"""
Context builder for Spectra.

Assembles the system prompt that is prepended to every model call.
Priority (highest first):
    1. Base personality
    2. Current PC activities
    3. Recent conversation history summary
    4. User profile highlights
    5. Time-of-day context
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4
# Leave room for model's max context; Qwen2-1.5B supports 32 k tokens but
# we keep things conservative to stay fast.
_MAX_CONTEXT_CHARS = 2000


class ContextBuilder:
    """Builds dynamic system prompts from memory and live monitoring data."""

    def __init__(self, config: Dict, database) -> None:
        """Initialise the builder.

        Args:
            config: Full parsed config dictionary.
            database: A :class:`spectra.memory.database.Database` instance.
        """
        self.config = config
        self.db = database
        self.companion_name: str = config.get("companion", {}).get("name", "Spectra")
        self.personality: str = config.get("companion", {}).get(
            "personality", "friendly, curious, witty, concise"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        current_activities: Optional[Dict] = None,
        extra_notes: Optional[str] = None,
    ) -> str:
        """Assemble and return the full system prompt string.

        Args:
            current_activities: Dict produced by the monitoring modules,
                e.g. ``{"spotify": "Artist - Song", "process": "gaming"}``.
            extra_notes: Any additional free-text to append at the end.

        Returns:
            The complete system prompt, trimmed to fit within context limits.
        """
        parts: List[str] = []

        parts.append(self._base_personality())

        time_ctx = self._time_context()
        if time_ctx:
            parts.append(time_ctx)

        if current_activities:
            activity_ctx = self._activity_context(current_activities)
            if activity_ctx:
                parts.append(activity_ctx)

        profile_ctx = self._profile_context()
        if profile_ctx:
            parts.append(profile_ctx)

        if extra_notes:
            parts.append(extra_notes)

        prompt = "\n\n".join(parts)

        # Trim to budget
        if len(prompt) > _MAX_CONTEXT_CHARS:
            prompt = prompt[:_MAX_CONTEXT_CHARS] + "\n[Context truncated to fit model limit]"

        return prompt

    def build_history(self, n: int = 10, session_id: Optional[str] = None) -> List[Dict]:
        """Return recent conversation history formatted for the model.

        Args:
            n: Number of recent turns to include.
            session_id: Restrict to a specific session if provided.

        Returns:
            List of ``{"role": …, "content": …}`` dicts.
        """
        turns = self.db.get_recent_conversations(n=n, session_id=session_id)
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _base_personality(self) -> str:
        """Return the fixed personality description block."""
        return (
            f"You are {self.companion_name}, a personal AI companion running locally on "
            f"the user's PC. Your personality: {self.personality}. "
            "Keep responses concise (1-3 sentences unless asked for detail). "
            "You have awareness of the user's PC activity and music listening habits "
            "and can reference them naturally."
        )

    def _time_context(self) -> str:
        """Return a short time-of-day description."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"
        return f"Current time: {datetime.now().strftime('%H:%M')} ({period})."

    def _activity_context(self, activities: Dict) -> str:
        """Convert the live activity dict into readable text.

        Args:
            activities: Mapping of activity type to description string.

        Returns:
            Formatted activity paragraph, or empty string.
        """
        lines: List[str] = []
        if activities.get("spotify"):
            lines.append(f"Currently listening to: {activities['spotify']}.")
        if activities.get("browser"):
            lines.append(f"Browser activity: {activities['browser']}.")
        if activities.get("process"):
            lines.append(f"Active application category: {activities['process']}.")
        if activities.get("download"):
            lines.append(f"Recent download: {activities['download']}.")
        if not lines:
            return ""
        return "Current PC activity:\n" + "\n".join(f"  • {l}" for l in lines)

    def _profile_context(self) -> str:
        """Build a short summary from the pc_profile table.

        Returns:
            Formatted profile paragraph, or empty string.
        """
        try:
            profile_rows = self.db.get_profile()
        except Exception as exc:
            logger.warning("Could not load profile: %s", exc)
            return ""

        if not profile_rows:
            return ""

        # Group by category and pick the 2 most recent per category
        by_cat: Dict[str, List[str]] = {}
        for row in profile_rows:
            cat = row["category"]
            by_cat.setdefault(cat, []).append(f"{row['key']}: {row['value']}")

        lines: List[str] = []
        for cat, items in list(by_cat.items())[:5]:
            lines.append(f"  {cat}: " + "; ".join(items[:2]))

        if not lines:
            return ""
        return "User profile highlights:\n" + "\n".join(lines)
