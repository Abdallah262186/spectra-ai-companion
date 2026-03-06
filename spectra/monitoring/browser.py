"""Browser activity monitor.

Detects Opera GX (and other common browsers) via process name and attempts to
read the active tab's title from the window title bar.  Known streaming
services (YouTube, Netflix, Twitch, etc.) are flagged specially.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 15  # seconds

# Map of lowercase substrings found in window titles to a friendly label
_STREAMING_SERVICES = {
    "youtube.com": "YouTube",
    "netflix.com": "Netflix",
    "twitch.tv": "Twitch",
    "spotify.com": "Spotify Web",
    "hulu.com": "Hulu",
    "disneyplus.com": "Disney+",
    "primevideo.com": "Prime Video",
}

# Known browser process names
_BROWSER_PROCESSES = {
    "opera.exe": "Opera GX",
    "operagx.exe": "Opera GX",
    "chrome.exe": "Chrome",
    "firefox.exe": "Firefox",
    "msedge.exe": "Edge",
    "brave.exe": "Brave",
}


class BrowserMonitor(threading.Thread):
    """Background thread that monitors browser activity.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any) -> None:
        super().__init__(daemon=True, name="BrowserMonitor")
        self.config = config
        self.db = db
        self._stop_event = threading.Event()
        self._last_title: Optional[str] = None

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Poll browser window titles repeatedly until stopped."""
        logger.info("BrowserMonitor started.")
        while not self._stop_event.is_set():
            try:
                self._check_browsers()
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("BrowserMonitor error: %s", exc)
            self._stop_event.wait(_POLL_INTERVAL)
        logger.info("BrowserMonitor stopped.")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _check_browsers(self) -> None:
        """Detect active browser and log significant title changes."""
        import psutil  # noqa: PLC0415

        active_browser: Optional[str] = None
        for proc in psutil.process_iter(["name"]):
            pname = proc.info["name"].lower()
            if pname in _BROWSER_PROCESSES:
                active_browser = _BROWSER_PROCESSES[pname]
                break

        if not active_browser:
            self._last_title = None
            return

        title = self._get_active_window_title()
        if not title or title == self._last_title:
            return

        self._last_title = title

        # Check for streaming services
        for fragment, label in _STREAMING_SERVICES.items():
            if fragment in title.lower():
                self.db.save_activity(
                    "browser",
                    f"Watching/listening on {label} via {active_browser}: {title[:80]}",
                )
                logger.debug("Browser streaming: %s on %s", label, active_browser)
                return

        # Generic tab change
        self.db.save_activity("browser", f"{active_browser}: {title[:80]}")
        logger.debug("Browser: %s – %s", active_browser, title[:80])

    @staticmethod
    def _get_active_window_title() -> str:
        """Return the foreground window's title text.

        Returns:
            Window title string, or empty string if unavailable.
        """
        try:
            import win32gui  # noqa: PLC0415

            hwnd = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(hwnd)
        except Exception:  # noqa: BLE001
            try:
                import pygetwindow as gw  # noqa: PLC0415

                active = gw.getActiveWindow()
                return active.title if active else ""
            except Exception:  # noqa: BLE001
                return ""
