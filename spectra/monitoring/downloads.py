"""
Downloads folder watcher.

Uses the ``watchdog`` library to monitor the configured Downloads
directory for new files and categorises them by extension.
"""

import logging
import os
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Extension → category mapping
EXTENSION_CATEGORIES = {
    # Images
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".bmp": "image", ".webp": "image", ".svg": "image", ".ico": "image",
    # Videos
    ".mp4": "video", ".mkv": "video", ".avi": "video", ".mov": "video",
    ".wmv": "video", ".flv": "video", ".webm": "video",
    # Audio
    ".mp3": "audio", ".flac": "audio", ".wav": "audio", ".aac": "audio",
    ".ogg": "audio", ".m4a": "audio",
    # Documents
    ".pdf": "document", ".docx": "document", ".doc": "document",
    ".xlsx": "document", ".xls": "document", ".pptx": "document",
    ".txt": "document", ".csv": "document",
    # Archives
    ".zip": "archive", ".rar": "archive", ".7z": "archive",
    ".tar": "archive", ".gz": "archive",
    # Executables / installers
    ".exe": "executable", ".msi": "executable", ".apk": "executable",
    # Code
    ".py": "code", ".js": "code", ".ts": "code", ".java": "code",
    ".cpp": "code", ".c": "code", ".cs": "code", ".go": "code",
}


def categorise_file(filepath: str) -> str:
    """Return a category label for a file based on its extension.

    Args:
        filepath: Full file path.

    Returns:
        Category string, or ``"other"`` for unknown extensions.
    """
    ext = Path(filepath).suffix.lower()
    return EXTENSION_CATEGORIES.get(ext, "other")


class DownloadEventHandler:
    """Watchdog event handler that logs new files to the database."""

    def __init__(self, database=None, on_new_file: Optional[Callable[[str], None]] = None) -> None:
        """Initialise the handler.

        Args:
            database: Optional database for activity logging.
            on_new_file: Optional callback invoked with the new file path.
        """
        self.db = database
        self.on_new_file = on_new_file

    def on_created(self, event) -> None:
        """Handle a file creation event from watchdog.

        Args:
            event: Watchdog ``FileCreatedEvent``.
        """
        if event.is_directory:
            return
        path = event.src_path
        size = 0
        try:
            size = os.path.getsize(path)
        except OSError:
            pass

        category = categorise_file(path)
        name = Path(path).name
        detail = f"New {category} download: {name} ({size:,} bytes)"
        logger.info(detail)

        if self.db:
            try:
                self.db.save_activity("download", detail)
            except Exception as exc:
                logger.warning("DB log error: %s", exc)

        if self.on_new_file:
            try:
                self.on_new_file(path)
            except Exception as exc:
                logger.warning("on_new_file callback error: %s", exc)


class DownloadsMonitor:
    """Monitors a Downloads folder for new files using watchdog."""

    def __init__(self, config: dict, database=None) -> None:
        """Initialise the monitor.

        Args:
            config: Full parsed config dictionary.
            database: Optional database for activity logging.
        """
        downloads_path = config.get("monitoring", {}).get("downloads_path", "~/Downloads")
        self.watch_path = str(Path(downloads_path).expanduser())
        self.db = database
        self._observer = None

    def start(self) -> None:
        """Start the watchdog observer thread."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _Handler(FileSystemEventHandler):
                def __init__(self, delegate):
                    self._d = delegate

                def on_created(self, event):
                    self._d.on_created(event)

            handler = DownloadEventHandler(database=self.db)
            wrapper = _Handler(handler)

            self._observer = Observer()
            self._observer.schedule(wrapper, self.watch_path, recursive=False)
            self._observer.daemon = True
            self._observer.start()
            logger.info("Watching Downloads folder: %s", self.watch_path)
        except Exception as exc:
            logger.error("Could not start Downloads monitor: %s", exc)

    def stop(self) -> None:
        """Stop the watchdog observer thread."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
        logger.info("Downloads monitor stopped.")

    def get_status(self) -> Optional[str]:
        """Return the most recent download event from the database, if any.

        Returns:
            Short description string, or None.
        """
        if not self.db:
            return None
        try:
            events = self.db.get_recent_activities(n=5)
            for event in events:
                if event.get("activity_type") == "download":
                    return event.get("details", "")
        except Exception:
            pass
        return None
