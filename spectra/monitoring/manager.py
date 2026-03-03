"""
Monitoring coordinator.

Aggregates Spotify, browser, downloads, and process monitors and
exposes a single ``get_current_activities()`` method used by the
context builder and proactive system.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MonitoringManager:
    """Coordinates all monitoring sub-modules."""

    def __init__(self, config: Dict, database) -> None:
        """Initialise and start all configured monitoring modules.

        Args:
            config: Full parsed config dictionary.
            database: :class:`spectra.memory.database.Database` instance.
        """
        self.config = config
        self.db = database
        monitoring_cfg = config.get("monitoring", {})

        self.spotify_monitor = None
        self.browser_monitor = None
        self.downloads_monitor = None
        self.process_monitor = None

        if monitoring_cfg.get("spotify", True):
            try:
                from spectra.monitoring.spotify import SpotifyMonitor
                self.spotify_monitor = SpotifyMonitor(database=database)
                logger.info("Spotify monitor initialised.")
            except Exception as exc:
                logger.warning("Spotify monitor failed to initialise: %s", exc)

        if monitoring_cfg.get("browser", True):
            try:
                from spectra.monitoring.browser import BrowserMonitor
                self.browser_monitor = BrowserMonitor(database=database)
                logger.info("Browser monitor initialised.")
            except Exception as exc:
                logger.warning("Browser monitor failed to initialise: %s", exc)

        if monitoring_cfg.get("downloads", True):
            try:
                from spectra.monitoring.downloads import DownloadsMonitor
                self.downloads_monitor = DownloadsMonitor(config=config, database=database)
                self.downloads_monitor.start()
                logger.info("Downloads monitor started.")
            except Exception as exc:
                logger.warning("Downloads monitor failed to start: %s", exc)

        if monitoring_cfg.get("processes", True):
            try:
                from spectra.monitoring.processes import ProcessMonitor
                self.process_monitor = ProcessMonitor(database=database)
                logger.info("Process monitor initialised.")
            except Exception as exc:
                logger.warning("Process monitor failed to initialise: %s", exc)

    def get_current_activities(self) -> Optional[Dict]:
        """Collect and return the current state of all monitors.

        Returns:
            Dict with keys ``spotify``, ``browser``, ``process``, ``download``,
            each containing a short status string or None.
        """
        activities: Dict[str, Optional[str]] = {
            "spotify": None,
            "browser": None,
            "process": None,
            "download": None,
        }

        if self.spotify_monitor:
            try:
                activities["spotify"] = self.spotify_monitor.get_status()
            except Exception as exc:
                logger.debug("Spotify status error: %s", exc)

        if self.browser_monitor:
            try:
                activities["browser"] = self.browser_monitor.get_status()
            except Exception as exc:
                logger.debug("Browser status error: %s", exc)

        if self.process_monitor:
            try:
                self.process_monitor.check_and_log_changes()
                activities["process"] = self.process_monitor.get_status()
            except Exception as exc:
                logger.debug("Process status error: %s", exc)

        if self.downloads_monitor:
            try:
                activities["download"] = self.downloads_monitor.get_status()
            except Exception as exc:
                logger.debug("Downloads status error: %s", exc)

        return activities

    def stop(self) -> None:
        """Stop all background monitoring threads."""
        if self.downloads_monitor:
            try:
                self.downloads_monitor.stop()
            except Exception as exc:
                logger.warning("Error stopping downloads monitor: %s", exc)
