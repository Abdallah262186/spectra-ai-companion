"""Downloads folder monitor.

Uses the ``watchdog`` library to observe the configured Downloads directory for
new files and logs each detected file to the activity database.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Maps file extension groups to a human-friendly category label
_CATEGORIES: Dict[str, str] = {
    # Images
    "jpg": "image", "jpeg": "image", "png": "image", "gif": "image",
    "webp": "image", "bmp": "image", "svg": "image", "ico": "image",
    # Videos
    "mp4": "video", "mkv": "video", "avi": "video", "mov": "video",
    "wmv": "video", "flv": "video", "webm": "video",
    # Audio
    "mp3": "audio", "flac": "audio", "wav": "audio", "ogg": "audio",
    "m4a": "audio", "aac": "audio",
    # Documents
    "pdf": "document", "doc": "document", "docx": "document",
    "xls": "document", "xlsx": "document", "ppt": "document",
    "pptx": "document", "txt": "document", "csv": "document",
    # Archives
    "zip": "archive", "rar": "archive", "7z": "archive",
    "tar": "archive", "gz": "archive", "bz2": "archive",
    # Executables / installers
    "exe": "executable", "msi": "executable", "bat": "executable",
    "sh": "executable", "appimage": "executable",
    # Code
    "py": "code", "js": "code", "ts": "code", "java": "code",
    "cpp": "code", "c": "code", "h": "code", "cs": "code",
}


def _categorise(filename: str) -> str:
    """Return the category label for a given filename.

    Args:
        filename: Bare filename (with or without path component).

    Returns:
        Category string such as ``"image"`` or ``"other"``.
    """
    ext = Path(filename).suffix.lstrip(".").lower()
    return _CATEGORIES.get(ext, "other")


class DownloadsMonitor(threading.Thread):
    """Watches the Downloads folder using ``watchdog`` and logs new files.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any) -> None:
        super().__init__(daemon=True, name="DownloadsMonitor")
        self.config = config
        self.db = db
        raw_path = config.get("monitoring", {}).get("downloads_path", "~/Downloads")
        self.downloads_path = Path(os.path.expanduser(raw_path))
        self._stop_event = threading.Event()
        self._observer = None

    def stop(self) -> None:
        """Stop the watchdog observer and the thread."""
        self._stop_event.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Observer stop error: %s", exc)

    def run(self) -> None:
        """Start the watchdog observer and block until stopped."""
        try:
            from watchdog.observers import Observer  # noqa: PLC0415
            from watchdog.events import FileSystemEventHandler, FileCreatedEvent  # noqa: PLC0415
        except ImportError:
            logger.warning("watchdog not installed; DownloadsMonitor disabled.")
            return

        if not self.downloads_path.exists():
            logger.warning("Downloads path does not exist: %s", self.downloads_path)
            return

        db = self.db

        class _Handler(FileSystemEventHandler):
            def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
                if event.is_directory:
                    return
                filename = Path(event.src_path).name
                category = _categorise(filename)
                size = 0
                try:
                    size = Path(event.src_path).stat().st_size
                except OSError:
                    pass
                details = f"New {category}: {filename} ({size:,} bytes)"
                db.save_activity("download", details)
                logger.info("Download detected: %s", details)

        self._observer = Observer()
        self._observer.schedule(_Handler(), str(self.downloads_path), recursive=False)
        self._observer.start()
        logger.info("DownloadsMonitor watching: %s", self.downloads_path)

        self._stop_event.wait()
        logger.info("DownloadsMonitor stopped.")
