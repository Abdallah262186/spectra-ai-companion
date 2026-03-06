"""Spotify monitor – detects currently playing track via window title.

On Windows, Spotify embeds the track info in its window title as
``"Artist - Song"`` when music is playing, or ``"Spotify"`` / ``"Spotify Free"``
when idle.  This module polls that title at a configurable interval and logs
changes to the activity database.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 10  # seconds


class SpotifyMonitor(threading.Thread):
    """Background thread that tracks Spotify playback.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any) -> None:
        super().__init__(daemon=True, name="SpotifyMonitor")
        self.config = config
        self.db = db
        self._stop_event = threading.Event()
        self._last_track: Optional[str] = None

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Poll Spotify window title repeatedly until stopped."""
        logger.info("SpotifyMonitor started.")
        while not self._stop_event.is_set():
            try:
                self._check_spotify()
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("SpotifyMonitor error: %s", exc)
            self._stop_event.wait(_POLL_INTERVAL)
        logger.info("SpotifyMonitor stopped.")

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _check_spotify(self) -> None:
        """Detect Spotify state and log any track change."""
        import psutil  # noqa: PLC0415

        spotify_running = any(
            p.name().lower() in ("spotify.exe", "spotify")
            for p in psutil.process_iter(["name"])
        )
        if not spotify_running:
            if self._last_track is not None:
                self._last_track = None
                self.db.save_activity("spotify", "Spotify closed")
            return

        title = self._get_spotify_window_title()
        if title and " - " in title and title not in ("Spotify", "Spotify Free", "Spotify Premium"):
            if title != self._last_track:
                self._last_track = title
                self.db.save_activity("spotify", f"Now playing: {title}")
                logger.debug("Spotify: %s", title)
        elif title in ("Spotify", "Spotify Free", "Spotify Premium", ""):
            # Paused / idle
            if self._last_track is not None:
                self._last_track = None
                self.db.save_activity("spotify", "Spotify paused")

    @staticmethod
    def _get_spotify_window_title() -> str:
        """Return the Spotify window title, or empty string if unavailable.

        Returns:
            Window title string.
        """
        try:
            import win32gui  # noqa: PLC0415

            titles: list[str] = []

            def _cb(hwnd: int, _: Any) -> bool:
                text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                if "chrome_widget_win" in class_name.lower() and text:
                    titles.append(text)
                return True

            win32gui.EnumWindows(_cb, None)
            for t in titles:
                if "spotify" in t.lower() or " - " in t:
                    return t
            return ""
        except Exception:  # noqa: BLE001
            # Fallback: try pygetwindow
            try:
                import pygetwindow as gw  # noqa: PLC0415

                windows = gw.getWindowsWithTitle("Spotify")
                if windows:
                    return windows[0].title
            except Exception:  # noqa: BLE001
                pass
            return ""

    def get_current_track(self) -> Optional[str]:
        """Return the currently playing track, or None if nothing is playing.

        Returns:
            Track string (``"Artist - Song"``), or None.
        """
        return self._last_track
