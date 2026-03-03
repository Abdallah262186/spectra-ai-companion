"""
Spotify monitoring module.

Detects whether Spotify is running and, on Windows, reads the window
title to extract the currently playing artist and track.
"""

import logging
import platform
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


def _get_spotify_window_title() -> Optional[str]:
    """Attempt to read Spotify's window title on Windows.

    Returns:
        The raw window title string, or None if unavailable.
    """
    if platform.system() != "Windows":
        return None
    try:
        import win32gui  # type: ignore

        def _callback(hwnd, titles):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and "Spotify" in win32gui.GetClassName(hwnd):
                    titles.append(title)

        titles = []
        win32gui.EnumWindows(_callback, titles)
        return titles[0] if titles else None
    except Exception:
        # Fall back to pygetwindow
        try:
            import pygetwindow as gw  # type: ignore

            wins = gw.getWindowsWithTitle("Spotify")
            if wins:
                return wins[0].title
        except Exception:
            pass
    return None


class SpotifyMonitor:
    """Detects Spotify playback via process inspection and window title parsing."""

    PROCESS_NAMES = {"Spotify.exe", "spotify"}

    def __init__(self, database=None) -> None:
        """Initialise the monitor.

        Args:
            database: Optional :class:`spectra.memory.database.Database` for
                logging activity events.
        """
        self.db = database
        self._last_track: Optional[str] = None

    def is_running(self) -> bool:
        """Return True if a Spotify process is currently running."""
        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] in self.PROCESS_NAMES:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False

    def get_current_track(self) -> Optional[str]:
        """Return ``"Artist - Song"`` if Spotify is playing, else None.

        Falls back to just ``"Spotify"`` if the window title cannot be read.
        """
        if not self.is_running():
            return None

        title = _get_spotify_window_title()
        if title and " - " in title and title != "Spotify":
            track = title.strip()
            if track != self._last_track:
                self._last_track = track
                if self.db:
                    try:
                        self.db.save_activity("spotify", f"Now playing: {track}")
                    except Exception as exc:
                        logger.warning("DB log error: %s", exc)
            return track

        return "Spotify (track unknown)"

    def get_status(self) -> Optional[str]:
        """Return a human-readable status string for the context builder."""
        return self.get_current_track()
