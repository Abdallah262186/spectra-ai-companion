"""
Initial PC scanner.

Runs once on first launch (or on demand via ``/scan``) to build the
user's profile in the database.  Scans:
    • Configured folders for file-type counts
    • Windows registry for installed programs (Windows only)
    • Opera GX bookmarks file (Windows only)
"""

import json
import logging
import os
import platform
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Extension → category mapping (reused from downloads but simplified)
EXTENSION_CATEGORIES = {
    ".py": "code", ".js": "code", ".ts": "code", ".java": "code",
    ".cpp": "code", ".c": "code", ".cs": "code", ".go": "code",
    ".mp3": "audio", ".flac": "audio", ".wav": "audio", ".aac": "audio",
    ".mp4": "video", ".mkv": "video", ".avi": "video",
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".pdf": "document", ".docx": "document", ".txt": "document",
    ".zip": "archive", ".rar": "archive", ".7z": "archive",
    ".exe": "executable", ".msi": "executable",
}


class PCScanner:
    """Builds a user profile by inspecting folders, registry, and bookmarks."""

    def __init__(self, config: Dict, database) -> None:
        """Initialise the scanner.

        Args:
            config: Full parsed config dictionary.
            database: :class:`spectra.memory.database.Database` instance.
        """
        self.config = config
        self.db = database
        scan_cfg = config.get("scan", {})
        raw_folders: List[str] = scan_cfg.get("folders", ["~/Documents", "~/Downloads"])
        self.folders = [str(Path(f).expanduser()) for f in raw_folders]
        self.scan_registry: bool = scan_cfg.get("scan_registry", True)
        self.scan_bookmarks: bool = scan_cfg.get("scan_bookmarks", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all scan phases and persist results to the database."""
        logger.info("Starting PC scan …")
        self._scan_folders()
        if self.scan_registry and platform.system() == "Windows":
            self._scan_registry()
        if self.scan_bookmarks and platform.system() == "Windows":
            self._scan_bookmarks()
        logger.info("PC scan complete.")

    # ------------------------------------------------------------------
    # Folder scan
    # ------------------------------------------------------------------

    def _scan_folders(self) -> None:
        """Count files by category in each configured folder (non-recursive)."""
        for folder in self.folders:
            if not os.path.isdir(folder):
                logger.debug("Scan folder does not exist: %s", folder)
                continue
            counts: Dict[str, int] = {}
            try:
                for entry in os.scandir(folder):
                    if entry.is_file(follow_symlinks=False):
                        ext = Path(entry.name).suffix.lower()
                        cat = EXTENSION_CATEGORIES.get(ext, "other")
                        counts[cat] = counts.get(cat, 0) + 1
            except PermissionError as exc:
                logger.warning("Permission denied scanning %s: %s", folder, exc)
                continue

            folder_label = Path(folder).name
            for cat, count in counts.items():
                self.db.update_profile("files", f"{folder_label}_{cat}", str(count))
                logger.debug("  %s/%s: %d files", folder_label, cat, count)

    # ------------------------------------------------------------------
    # Registry scan (Windows only)
    # ------------------------------------------------------------------

    def _scan_registry(self) -> None:
        """Read installed program names from the Windows registry."""
        try:
            import winreg  # type: ignore

            uninstall_key = (
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
            )
            programs: List[str] = []
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    key = winreg.OpenKey(hive, uninstall_key)
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            sub_key_name = winreg.EnumKey(key, i)
                            sub_key = winreg.OpenKey(key, sub_key_name)
                            try:
                                name, _ = winreg.QueryValueEx(sub_key, "DisplayName")
                                if name:
                                    programs.append(str(name))
                            except FileNotFoundError:
                                pass
                            winreg.CloseKey(sub_key)
                        except OSError:
                            pass
                    winreg.CloseKey(key)
                except OSError:
                    pass

            if programs:
                self.db.update_profile("software", "installed_programs", "; ".join(programs[:50]))
                logger.info("Registry scan: found %d programs.", len(programs))
        except ImportError:
            logger.debug("winreg not available — skipping registry scan.")
        except Exception as exc:
            logger.warning("Registry scan error: %s", exc)

    # ------------------------------------------------------------------
    # Bookmarks scan (Windows / Opera GX)
    # ------------------------------------------------------------------

    def _scan_bookmarks(self) -> None:
        """Parse Opera GX bookmark file and save domain list to profile."""
        opera_profile = Path(os.environ.get("APPDATA", "")) / "Opera Software" / "Opera GX Stable"
        bookmarks_file = opera_profile / "Bookmarks"

        if not bookmarks_file.exists():
            logger.debug("Opera GX Bookmarks file not found at %s", bookmarks_file)
            return

        try:
            with open(bookmarks_file, encoding="utf-8") as fh:
                data = json.load(fh)

            urls: List[str] = []
            self._extract_bookmark_urls(data.get("roots", {}), urls)

            if urls:
                self.db.update_profile("bookmarks", "bookmark_urls", "; ".join(urls[:100]))
                logger.info("Bookmarks scan: found %d bookmarks.", len(urls))
        except Exception as exc:
            logger.warning("Bookmarks scan error: %s", exc)

    def _extract_bookmark_urls(self, node, result: List[str]) -> None:
        """Recursively extract URLs from a Chromium bookmarks JSON node.

        Args:
            node: Current JSON node (dict or list).
            result: Accumulator list for URL strings.
        """
        if isinstance(node, dict):
            if node.get("type") == "url":
                url = node.get("url", "")
                if url:
                    result.append(url)
            for value in node.values():
                self._extract_bookmark_urls(value, result)
        elif isinstance(node, list):
            for item in node:
                self._extract_bookmark_urls(item, result)
