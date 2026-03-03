"""
Process monitor.

Uses ``psutil`` to list running processes, categorise them (games,
dev tools, media, browsers), and detect significant changes.
"""

import logging
from typing import Dict, Optional, Set

import psutil

logger = logging.getLogger(__name__)

# Known process names (lower-case) → category
PROCESS_CATEGORIES: Dict[str, str] = {
    # Games / launchers
    "steam.exe": "gaming",
    "epicgameslauncher.exe": "gaming",
    "gog galaxy.exe": "gaming",
    "battlenet.exe": "gaming",
    "leagueoflegends.exe": "gaming",
    "valorant.exe": "gaming",
    "csgo.exe": "gaming",
    "cyberpunk2077.exe": "gaming",
    "witcher3.exe": "gaming",
    "minecraft.exe": "gaming",
    "minecraft.java": "gaming",
    # Dev tools
    "code.exe": "development",
    "devenv.exe": "development",
    "pycharm64.exe": "development",
    "idea64.exe": "development",
    "webstorm64.exe": "development",
    "python.exe": "development",
    "python3": "development",
    "node.exe": "development",
    "git.exe": "development",
    "powershell.exe": "development",
    "cmd.exe": "development",
    "wt.exe": "development",
    # Media
    "vlc.exe": "media",
    "spotify.exe": "media",
    "discord.exe": "communication",
    "zoom.exe": "communication",
    "teams.exe": "communication",
    "slack.exe": "communication",
    # Browsers
    "opera.exe": "browsing",
    "opera_gx.exe": "browsing",
    "chrome.exe": "browsing",
    "firefox.exe": "browsing",
    "msedge.exe": "browsing",
}


class ProcessMonitor:
    """Tracks running processes and detects category changes."""

    def __init__(self, database=None) -> None:
        """Initialise the monitor.

        Args:
            database: Optional database for activity logging.
        """
        self.db = database
        self._prev_categories: Set[str] = set()

    def get_running_categories(self) -> Set[str]:
        """Return the set of categories for currently running processes.

        Returns:
            Set of category label strings.
        """
        categories: Set[str] = set()
        try:
            for proc in psutil.process_iter(["name"]):
                name = (proc.info.get("name") or "").lower()
                cat = PROCESS_CATEGORIES.get(name)
                if cat:
                    categories.add(cat)
        except Exception as exc:
            logger.debug("Process iteration error: %s", exc)
        return categories

    def check_and_log_changes(self) -> None:
        """Compare running categories to the previous snapshot and log changes."""
        current = self.get_running_categories()
        started = current - self._prev_categories
        stopped = self._prev_categories - current

        for cat in started:
            detail = f"Activity started: {cat}"
            logger.info(detail)
            if self.db:
                try:
                    self.db.save_activity("process", detail)
                except Exception as exc:
                    logger.warning("DB log error: %s", exc)

        for cat in stopped:
            detail = f"Activity stopped: {cat}"
            logger.info(detail)
            if self.db:
                try:
                    self.db.save_activity("process", detail)
                except Exception as exc:
                    logger.warning("DB log error: %s", exc)

        self._prev_categories = current

    def get_status(self) -> Optional[str]:
        """Return a comma-separated string of active categories, or None.

        Returns:
            Short description string, or None if nothing notable is running.
        """
        cats = self.get_running_categories()
        if not cats:
            return None
        return ", ".join(sorted(cats))
