"""Process monitor – detects running programs and categorises them.

Polls the process list at a regular interval and logs significant changes
(new processes started, known processes stopped) to the activity database.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 20  # seconds

# Known process names → category label
_KNOWN_PROCESSES: Dict[str, str] = {
    # Games
    "steam.exe": "game_launcher", "epicgameslauncher.exe": "game_launcher",
    "gog galaxy.exe": "game_launcher", "battlenet.exe": "game_launcher",
    "cs2.exe": "game", "dota2.exe": "game", "fortnite.exe": "game",
    "minecraft.exe": "game", "javaw.exe": "game",
    # Dev tools
    "code.exe": "dev", "pycharm64.exe": "dev", "idea64.exe": "dev",
    "devenv.exe": "dev", "rider64.exe": "dev", "sublime_text.exe": "dev",
    "git.exe": "dev", "python.exe": "dev", "node.exe": "dev",
    "cmd.exe": "terminal", "powershell.exe": "terminal",
    "windowsterminal.exe": "terminal",
    # Media
    "vlc.exe": "media", "mpv.exe": "media", "obs64.exe": "streaming",
    "obs32.exe": "streaming", "foobar2000.exe": "media",
    "aimp.exe": "media",
    # Browsers (already tracked by BrowserMonitor, but included for completeness)
    "opera.exe": "browser", "chrome.exe": "browser",
    "firefox.exe": "browser", "msedge.exe": "browser",
    # Communication
    "discord.exe": "communication", "slack.exe": "communication",
    "teams.exe": "communication", "zoom.exe": "communication",
    "telegram.exe": "communication",
    # Productivity
    "excel.exe": "productivity", "winword.exe": "productivity",
    "powerpnt.exe": "productivity", "onenote.exe": "productivity",
    "notepad.exe": "productivity", "notepad++.exe": "productivity",
}


class ProcessMonitor(threading.Thread):
    """Background thread that tracks running processes.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any) -> None:
        super().__init__(daemon=True, name="ProcessMonitor")
        self.config = config
        self.db = db
        self._stop_event = threading.Event()
        self._known_running: Set[str] = set()

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Poll processes repeatedly until stopped."""
        logger.info("ProcessMonitor started.")
        # Seed initial state without logging
        self._known_running = self._current_known_processes()

        while not self._stop_event.is_set():
            self._stop_event.wait(_POLL_INTERVAL)
            if self._stop_event.is_set():
                break
            try:
                self._check_processes()
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("ProcessMonitor error: %s", exc)

        logger.info("ProcessMonitor stopped.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_known_processes(self) -> Set[str]:
        """Return the set of currently running *known* process names.

        Returns:
            Set of lowercase process names that appear in ``_KNOWN_PROCESSES``.
        """
        try:
            import psutil  # noqa: PLC0415

            running = set()
            for proc in psutil.process_iter(["name"]):
                pname = proc.info["name"].lower()
                if pname in _KNOWN_PROCESSES:
                    running.add(pname)
            return running
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Process scan error: %s", exc)
            return set()

    def _check_processes(self) -> None:
        """Detect started / stopped known processes and log them."""
        current = self._current_known_processes()

        started = current - self._known_running
        stopped = self._known_running - current

        for pname in started:
            category = _KNOWN_PROCESSES[pname]
            self.db.save_activity("process", f"Started [{category}]: {pname}")
            logger.info("Process started: %s (%s)", pname, category)

        for pname in stopped:
            category = _KNOWN_PROCESSES[pname]
            self.db.save_activity("process", f"Stopped [{category}]: {pname}")
            logger.info("Process stopped: %s (%s)", pname, category)

        self._known_running = current

    def get_running_categories(self) -> Dict[str, str]:
        """Return a snapshot of currently running known processes and their categories.

        Returns:
            Dict mapping process name → category label.
        """
        return {p: _KNOWN_PROCESSES[p] for p in self._known_running}
