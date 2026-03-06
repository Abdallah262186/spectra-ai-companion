"""Initial PC scanner.

Runs once on first launch (or on demand via ``/scan``) to build a user profile
by inspecting:
- Configured filesystem folders (Documents, Downloads, Music) for file types.
- Windows registry for installed programs (``winreg``).
- Opera GX / Chromium-based browser bookmark files.

Results are stored in the ``pc_profile`` table.
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PCScanner:
    """Performs an initial PC scan and populates the user profile.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any) -> None:
        self.config = config
        self.db = db
        scan_cfg = config.get("scan", {})
        self.folders: List[str] = scan_cfg.get("folders", ["~/Documents", "~/Downloads"])
        self.scan_registry: bool = scan_cfg.get("scan_registry", True)
        self.scan_bookmarks: bool = scan_cfg.get("scan_bookmarks", True)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_if_needed(self) -> None:
        """Run the scan only if the profile table is empty."""
        existing = self.db.get_profile()
        if not existing:
            logger.info("No profile found – running initial PC scan.")
            self.run()
        else:
            logger.info("Profile already exists – skipping scan (use /scan to force).")

    def run(self) -> None:
        """Execute all scan phases."""
        logger.info("Starting PC scan…")
        self._scan_folders()
        if self.scan_registry:
            self._scan_registry()
        if self.scan_bookmarks:
            self._scan_bookmarks()
        logger.info("PC scan complete.")

    # ------------------------------------------------------------------
    # Folder scan
    # ------------------------------------------------------------------

    def _scan_folders(self) -> None:
        """Scan configured folders and summarise file types."""
        for folder_raw in self.folders:
            folder = Path(os.path.expanduser(folder_raw))
            if not folder.exists():
                logger.debug("Folder does not exist: %s", folder)
                continue
            ext_counter: Counter = Counter()
            file_count = 0
            try:
                for entry in folder.iterdir():
                    if entry.is_file():
                        ext_counter[entry.suffix.lower()] += 1
                        file_count += 1
            except PermissionError as exc:
                logger.debug("Permission denied for %s: %s", folder, exc)
                continue

            self.db.update_profile("files", f"{folder.name}_total", str(file_count))
            # Top 5 extensions
            for ext, count in ext_counter.most_common(5):
                key = f"{folder.name}_ext_{ext.lstrip('.') or 'no_ext'}"
                self.db.update_profile("files", key, str(count))
            logger.info("Scanned %s: %d files", folder, file_count)

    # ------------------------------------------------------------------
    # Registry scan
    # ------------------------------------------------------------------

    def _scan_registry(self) -> None:
        """Read HKLM and HKCU Uninstall keys to list installed programs."""
        try:
            import winreg  # noqa: PLC0415
        except ImportError:
            logger.debug("winreg not available (not on Windows).")
            return

        reg_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]

        programs: List[str] = []
        for hive, path in reg_paths:
            try:
                key = winreg.OpenKey(hive, path)
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        subkey = winreg.OpenKey(key, subkey_name)
                        try:
                            name, _ = winreg.QueryValueEx(subkey, "DisplayName")
                            if name and isinstance(name, str):
                                programs.append(name.strip())
                        except FileNotFoundError:
                            pass
                        finally:
                            winreg.CloseKey(subkey)
                    except Exception:  # noqa: BLE001
                        continue
                winreg.CloseKey(key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Registry read error for %s: %s", path, exc)

        unique_programs = sorted(set(programs))
        self.db.update_profile("software", "installed_count", str(len(unique_programs)))
        # Store up to 50 program names as a JSON list
        self.db.update_profile("software", "installed_list", json.dumps(unique_programs[:50]))
        logger.info("Registry scan: %d programs found.", len(unique_programs))

    # ------------------------------------------------------------------
    # Bookmark scan
    # ------------------------------------------------------------------

    def _scan_bookmarks(self) -> None:
        """Parse Chromium-based browser bookmarks (Opera GX and Chrome)."""
        bookmark_paths: List[Path] = []

        # Opera GX
        opera_gx = Path(os.path.expanduser(
            "~/AppData/Roaming/Opera Software/Opera GX Stable/Bookmarks"
        ))
        if opera_gx.exists():
            bookmark_paths.append(opera_gx)

        # Google Chrome
        chrome = Path(os.path.expanduser(
            "~/AppData/Local/Google/Chrome/User Data/Default/Bookmarks"
        ))
        if chrome.exists():
            bookmark_paths.append(chrome)

        all_bookmarks: List[str] = []
        for bp in bookmark_paths:
            all_bookmarks.extend(self._parse_bookmark_file(bp))

        if all_bookmarks:
            self.db.update_profile("browser", "bookmark_count", str(len(all_bookmarks)))
            # Store up to 30 bookmark titles
            self.db.update_profile("browser", "bookmarks_sample", json.dumps(all_bookmarks[:30]))
            logger.info("Bookmark scan: %d bookmarks found.", len(all_bookmarks))

    @staticmethod
    def _parse_bookmark_file(path: Path) -> List[str]:
        """Extract bookmark names from a Chromium Bookmarks JSON file.

        Args:
            path: Path to the Bookmarks file.

        Returns:
            List of bookmark name strings.
        """
        titles: List[str] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

            def _traverse(node: Any) -> None:
                if isinstance(node, dict):
                    if node.get("type") == "url":
                        name = node.get("name", "")
                        if name:
                            titles.append(name)
                    for child in node.get("children", []):
                        _traverse(child)

            _traverse(data.get("roots", {}))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Bookmark parse error for %s: %s", path, exc)
        return titles
