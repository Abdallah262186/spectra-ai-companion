"""
Browser monitoring module.

Detects running browsers (Opera GX, Chrome, Firefox, Edge) via process
names and attempts to read the active window title for tab context.
Recognises streaming services by keyword matching.
"""

import logging
import platform
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

# Map of process name → friendly browser name
BROWSER_PROCESSES = {
    "opera.exe": "Opera GX",
    "opera_gx.exe": "Opera GX",
    "chrome.exe": "Chrome",
    "firefox.exe": "Firefox",
    "msedge.exe": "Edge",
    "brave.exe": "Brave",
    "vivaldi.exe": "Vivaldi",
}

# Keywords in window titles that indicate streaming/content services
STREAMING_KEYWORDS = {
    "YouTube": "watching YouTube",
    "Netflix": "watching Netflix",
    "Twitch": "watching Twitch",
    "Prime Video": "watching Prime Video",
    "Disney+": "watching Disney+",
    "Spotify": "listening via browser",
}


def _get_window_titles_for_process(process_name: str):
    """Yield visible window titles whose owning process matches *process_name*.

    Args:
        process_name: Executable name (case-insensitive on Windows).

    Yields:
        Window title strings.
    """
    if platform.system() != "Windows":
        return
    try:
        import win32gui
        import win32process  # type: ignore

        target_pids = {
            p.pid
            for p in psutil.process_iter(["name"])
            if (p.info.get("name") or "").lower() == process_name.lower()
        }

        results = []

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid in target_pids:
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        results.append(title)

        win32gui.EnumWindows(_cb, None)
        yield from results
    except Exception as exc:
        logger.debug("Window title enumeration failed: %s", exc)


class BrowserMonitor:
    """Tracks browser activity and detects streaming service usage."""

    def __init__(self, database=None) -> None:
        """Initialise the monitor.

        Args:
            database: Optional :class:`spectra.memory.database.Database` for logging.
        """
        self.db = database
        self._last_status: Optional[str] = None

    def get_active_browser(self) -> Optional[str]:
        """Return the name of the first detected running browser, or None."""
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                for key, label in BROWSER_PROCESSES.items():
                    if name == key.lower():
                        return label
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def get_status(self) -> Optional[str]:
        """Return a description of current browser activity for the context builder.

        Returns:
            Short description string, or None if no browser is detected.
        """
        browser = self.get_active_browser()
        if not browser:
            return None

        # Try to get more detail from window titles on Windows
        for proc_name, label in BROWSER_PROCESSES.items():
            if label == browser:
                for title in _get_window_titles_for_process(proc_name):
                    for keyword, description in STREAMING_KEYWORDS.items():
                        if keyword.lower() in title.lower():
                            status = f"{description} ({browser})"
                            self._log_if_changed(status)
                            return status
                    # Return a trimmed page title
                    if title:
                        status = f"{browser}: {title[:60]}"
                        self._log_if_changed(status)
                        return status

        status = f"{browser} open"
        self._log_if_changed(status)
        return status

    def _log_if_changed(self, status: str) -> None:
        """Log the status to the database only when it changes.

        Args:
            status: Current activity description.
        """
        if status != self._last_status and self.db:
            self._last_status = status
            try:
                self.db.save_activity("browser", status)
            except Exception as exc:
                logger.warning("DB log error: %s", exc)
