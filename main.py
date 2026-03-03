"""
Spectra 2.0 — Personal AI Companion
Entry point.

Usage:
    python main.py [--skip-scan] [--no-proactive] [--train-now]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging setup (before any imports that use the logger)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("spectra.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("spectra.main")


def load_config(path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file.

    Args:
        path: Path to the config file (relative to CWD).

    Returns:
        Parsed config dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Spectra 2.0 — Personal AI Companion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip the initial PC scan on startup.",
    )
    parser.add_argument(
        "--no-proactive",
        action="store_true",
        help="Disable the proactive conversation system.",
    )
    parser.add_argument(
        "--train-now",
        action="store_true",
        help="Run LoRA fine-tuning immediately after startup.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Initialise all subsystems and start the interactive REPL."""
    args = parse_args()
    config = load_config(args.config)

    # Ensure the adapters directory exists
    adapter_path = config.get("training", {}).get("adapter_save_path", "adapters/")
    os.makedirs(adapter_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Database
    # -----------------------------------------------------------------------
    from spectra.memory.database import Database

    db_path = config.get("database", {}).get("path", "spectra_memory.db")
    logger.info("Opening database: %s", db_path)
    db = Database(db_path=db_path)

    # -----------------------------------------------------------------------
    # 2. Context builder
    # -----------------------------------------------------------------------
    from spectra.memory.context import ContextBuilder

    ctx = ContextBuilder(config=config, database=db)

    # -----------------------------------------------------------------------
    # 3. AI Engine
    # -----------------------------------------------------------------------
    from spectra.core.engine import AIEngine

    engine = AIEngine(config=config)
    logger.info("Loading AI model — this may take a minute …")
    try:
        engine.load()
    except Exception as exc:
        logger.critical("Failed to load AI model: %s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 4. Monitoring
    # -----------------------------------------------------------------------
    from spectra.monitoring.manager import MonitoringManager

    monitoring = MonitoringManager(config=config, database=db)

    # -----------------------------------------------------------------------
    # 5. Scanner (run once unless --skip-scan)
    # -----------------------------------------------------------------------
    from spectra.monitoring.scanner import PCScanner

    scanner = PCScanner(config=config, database=db)
    if not args.skip_scan:
        logger.info("Running initial PC scan …")
        try:
            scanner.run()
        except Exception as exc:
            logger.warning("PC scan failed (non-fatal): %s", exc)

    # -----------------------------------------------------------------------
    # 6. Trainer
    # -----------------------------------------------------------------------
    trainer = None
    if config.get("training", {}).get("enabled", True):
        from spectra.training.lora_trainer import LoRATrainer

        trainer = LoRATrainer(config=config, database=db)
        if args.train_now:
            logger.info("--train-now flag: starting training …")
            try:
                trainer.train()
            except Exception as exc:
                logger.error("Immediate training failed: %s", exc)

    # -----------------------------------------------------------------------
    # 7. Search
    # -----------------------------------------------------------------------
    searcher = None
    if config.get("search", {}).get("enabled", True):
        from spectra.search.web_search import WebSearcher

        searcher = WebSearcher(config=config)

    # -----------------------------------------------------------------------
    # 8. Conversation manager
    # -----------------------------------------------------------------------
    from spectra.core.conversation import ConversationManager

    conversation = ConversationManager(
        config=config,
        engine=engine,
        database=db,
        context_builder=ctx,
        monitoring_manager=monitoring,
        trainer=trainer,
        searcher=searcher,
        scanner=scanner,
    )

    # -----------------------------------------------------------------------
    # 9. Proactive system
    # -----------------------------------------------------------------------
    proactive = None
    proactive_cfg = config.get("proactive", {})
    if proactive_cfg.get("enabled", True) and not args.no_proactive:
        from spectra.core.proactive import ProactiveSystem

        proactive = ProactiveSystem(
            config=config,
            engine=engine,
            context_builder=ctx,
            database=db,
            inject_fn=conversation.inject_proactive_message,
            monitoring_manager=monitoring,
        )
        proactive.start()

    # -----------------------------------------------------------------------
    # 10. Start REPL
    # -----------------------------------------------------------------------
    try:
        conversation.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down …")
        if proactive:
            proactive.stop()
        monitoring.stop()
        db.close()
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()
