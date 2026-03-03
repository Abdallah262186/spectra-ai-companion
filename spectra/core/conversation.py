"""Interactive conversation manager.

Provides the main chat loop, command handling, and coloured terminal UI.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Colour constants (populated lazily so import never fails)
_CYAN = ""
_GREEN = ""
_YELLOW = ""
_RED = ""
_RESET = ""


def _init_colours() -> None:
    """Initialise colorama colours once."""
    global _CYAN, _GREEN, _YELLOW, _RED, _RESET  # noqa: PLW0603
    try:
        from colorama import Fore, Style, init as colorama_init  # noqa: PLC0415

        colorama_init(autoreset=False)
        _CYAN = Fore.CYAN
        _GREEN = Fore.GREEN
        _YELLOW = Fore.YELLOW
        _RED = Fore.RED
        _RESET = Style.RESET_ALL
    except ImportError:
        pass


_HELP_TEXT = """
Available commands:
  /help           – Show this help message
  /quit           – Exit Spectra
  /status         – Show current system status
  /train          – Run LoRA fine-tuning on conversation history
  /search <query> – Web search via DuckDuckGo
  /scan           – Re-run the initial PC scan
  /memory         – Show recent conversation summary
"""


class ConversationManager:
    """Manages the interactive chat loop and command dispatch.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
        engine: Loaded :class:`~spectra.core.engine.AIEngine` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any, engine: Any) -> None:
        self.config = config
        self.db = db
        self.engine = engine
        self.companion_name: str = config.get("companion", {}).get("name", "Spectra")
        self.session_id: str = str(uuid.uuid4())
        self.conversation_count: int = 0
        self.train_trigger: int = (
            config.get("training", {}).get("trigger_every_n_conversations", 50)
        )
        self.training_enabled: bool = config.get("training", {}).get("enabled", True)
        _init_colours()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enter the interactive chat loop (blocks until user quits)."""
        print(
            f"\n{_CYAN}{'=' * 60}{_RESET}\n"
            f"{_CYAN}  Spectra 2.0 – Personal AI Companion{_RESET}\n"
            f"{_CYAN}{'=' * 60}{_RESET}\n"
            f"Type {_YELLOW}/help{_RESET} for available commands.\n"
        )

        while True:
            try:
                user_input = input(f"{_GREEN}You > {_RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_exit = self._handle_command(user_input)
                if should_exit:
                    break
                continue

            self._handle_chat(user_input)

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    def _handle_command(self, raw: str) -> bool:
        """Dispatch a slash-command.

        Args:
            raw: Raw input string starting with ``/``.

        Returns:
            True if the application should exit, False otherwise.
        """
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/quit":
            print(f"{_CYAN}Goodbye!{_RESET}")
            return True

        if cmd == "/help":
            print(_HELP_TEXT)

        elif cmd == "/status":
            self._cmd_status()

        elif cmd == "/train":
            self._cmd_train()

        elif cmd == "/search":
            if not arg:
                print(f"{_YELLOW}Usage: /search <query>{_RESET}")
            else:
                self._cmd_search(arg)

        elif cmd == "/scan":
            self._cmd_scan()

        elif cmd == "/memory":
            self._cmd_memory()

        else:
            print(f"{_RED}Unknown command: {cmd}. Type /help for a list.{_RESET}")

        return False

    def _cmd_status(self) -> None:
        """Print a brief system status report."""
        print(f"\n{_CYAN}--- Spectra Status ---{_RESET}")
        print(f"  Session ID      : {self.session_id}")
        print(f"  Conversations   : {self.conversation_count} this session")
        print(f"  Model           : {self.engine.model_name}")
        print(f"  Training enabled: {self.training_enabled}")
        try:
            recent = self.db.get_recent_conversations(1)
            if recent:
                ts = recent[0].get("timestamp", "?")
                print(f"  Last message at : {ts}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Status error: %s", exc)
        print()

    def _cmd_train(self) -> None:
        """Trigger LoRA fine-tuning manually."""
        print(f"{_YELLOW}Starting LoRA training…{_RESET}")
        try:
            from spectra.training.lora_trainer import LoRATrainer  # noqa: PLC0415

            trainer = LoRATrainer(self.config, self.db)
            trainer.train()
            print(f"{_GREEN}Training complete.{_RESET}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Training failed: %s", exc)
            print(f"{_RED}Training failed: {exc}{_RESET}")

    def _cmd_search(self, query: str) -> None:
        """Run a DuckDuckGo web search and print results.

        Args:
            query: Search query string.
        """
        print(f"{_YELLOW}Searching for: {query}{_RESET}")
        try:
            from spectra.search.web_search import WebSearch  # noqa: PLC0415

            searcher = WebSearch(self.config)
            results = searcher.search(query)
            if results:
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    url = r.get("href", r.get("url", ""))
                    body = r.get("body", "")
                    print(f"\n{_CYAN}[{i}] {title}{_RESET}")
                    if url:
                        print(f"    {url}")
                    if body:
                        print(f"    {body[:200]}")
            else:
                print(f"{_YELLOW}No results found.{_RESET}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Search failed: %s", exc)
            print(f"{_RED}Search failed: {exc}{_RESET}")

    def _cmd_scan(self) -> None:
        """Re-run the PC scanner."""
        print(f"{_YELLOW}Scanning PC…{_RESET}")
        try:
            from spectra.monitoring.scanner import PCScanner  # noqa: PLC0415

            scanner = PCScanner(self.config, self.db)
            scanner.run()
            print(f"{_GREEN}Scan complete.{_RESET}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Scan failed: %s", exc)
            print(f"{_RED}Scan failed: {exc}{_RESET}")

    def _cmd_memory(self) -> None:
        """Display recent conversation history."""
        try:
            rows = self.db.get_recent_conversations(10)
            if not rows:
                print(f"{_YELLOW}No conversation history yet.{_RESET}")
                return
            print(f"\n{_CYAN}--- Recent Memory (last {len(rows)} turns) ---{_RESET}")
            for row in rows:
                role = row.get("role", "?")
                content = row.get("content", "")
                ts = row.get("timestamp", "")
                label = "You" if role == "user" else self.companion_name
                colour = _GREEN if role == "user" else _CYAN
                print(f"{colour}[{ts}] {label}: {content[:120]}{_RESET}")
            print()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Memory command failed: %s", exc)
            print(f"{_RED}Could not retrieve memory: {exc}{_RESET}")

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _handle_chat(self, user_input: str) -> None:
        """Process a normal user message and generate an AI response.

        Args:
            user_input: The text entered by the user.
        """
        # Save user message
        self.db.save_message(
            role="user",
            content=user_input,
            session_id=self.session_id,
        )

        # Build prompt messages
        from spectra.memory.context import ContextBuilder  # noqa: PLC0415

        builder = ContextBuilder(self.config, self.db)
        messages = builder.build_messages(user_input)

        # Print AI prefix then stream response
        print(f"\n{_CYAN}{self.companion_name} > {_RESET}", end="", flush=True)
        try:
            response = self.engine.generate_response(messages, stream=True)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Generation error: %s", exc)
            response = "I encountered an error. Please try again."
            print(response, end="")
        print("\n")

        # Save assistant message
        self.db.save_message(
            role="assistant",
            content=response,
            session_id=self.session_id,
        )

        self.conversation_count += 1

        # Auto-training trigger
        if (
            self.training_enabled
            and self.conversation_count > 0
            and self.conversation_count % self.train_trigger == 0
        ):
            logger.info("Auto-training trigger reached (%d conversations).", self.conversation_count)
            self._cmd_train()
