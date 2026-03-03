"""Spectra 2.0 - Personal AI Companion.

Entry point that initialises all subsystems and starts the interactive
chat loop.  Supports a handful of CLI flags for quick overrides.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging – file + stderr
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("spectra.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("spectra")


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Spectra 2.0 – Personal AI Companion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip the initial PC scan on first launch.",
    )
    parser.add_argument(
        "--no-proactive",
        action="store_true",
        help="Disable proactive conversation initiator.",
    )
    parser.add_argument(
        "--train-now",
        action="store_true",
        help="Run LoRA fine-tuning immediately on startup, then enter chat.",
    )
    return parser.parse_args()


def main() -> None:
    """Initialise all Spectra subsystems and start the chat loop."""
    args = parse_args()
    config = load_config(args.config)

    logger.info("Starting Spectra 2.0")

    # 1. Database
    from spectra.memory.database import Database  # noqa: PLC0415

    db = Database(config["database"]["path"])
    db.initialize()

    # 2. AI engine
    from spectra.core.engine import AIEngine  # noqa: PLC0415

    engine = AIEngine(config)
    engine.load_model()

    # 3. PC scan (first run or explicit request)
    if not args.skip_scan:
        from spectra.monitoring.scanner import PCScanner  # noqa: PLC0415

        scanner = PCScanner(config, db)
        scanner.run_if_needed()

    # 4. Monitoring threads
    monitor_threads = []
    from spectra.monitoring.processes import ProcessMonitor  # noqa: PLC0415
    from spectra.monitoring.downloads import DownloadsMonitor  # noqa: PLC0415
    from spectra.monitoring.spotify import SpotifyMonitor  # noqa: PLC0415
    from spectra.monitoring.browser import BrowserMonitor  # noqa: PLC0415

    if config["monitoring"].get("processes", True):
        pm = ProcessMonitor(config, db)
        pm.start()
        monitor_threads.append(pm)

    if config["monitoring"].get("downloads", True):
        dm = DownloadsMonitor(config, db)
        dm.start()
        monitor_threads.append(dm)

    if config["monitoring"].get("spotify", True):
        sm = SpotifyMonitor(config, db)
        sm.start()
        monitor_threads.append(sm)

    if config["monitoring"].get("browser", True):
        bm = BrowserMonitor(config, db)
        bm.start()
        monitor_threads.append(bm)

    # 5. Optional immediate training
    if args.train_now:
        from spectra.training.lora_trainer import LoRATrainer  # noqa: PLC0415

        trainer = LoRATrainer(config, db)
        trainer.train()

    # 6. Proactive system
    proactive_thread = None
    if not args.no_proactive and config["proactive"].get("enabled", True):
        from spectra.core.proactive import ProactiveSystem  # noqa: PLC0415

        proactive_thread = ProactiveSystem(config, db, engine)
        proactive_thread.start()

    # 7. Interactive chat loop
    from spectra.core.conversation import ConversationManager  # noqa: PLC0415

    manager = ConversationManager(config, db, engine)
    try:
        manager.start()
    except KeyboardInterrupt:
        logger.info("Shutting down Spectra…")
    finally:
        # Graceful shutdown
        for thread in monitor_threads:
            thread.stop()
        if proactive_thread is not None:
            proactive_thread.stop()
        logger.info("Goodbye!")


if __name__ == "__main__":
    main()
