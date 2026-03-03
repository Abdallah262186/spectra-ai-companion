"""
Interactive conversation loop for Spectra.

Provides a colored terminal REPL with slash-commands and tracks
the number of turns for the auto-training trigger.
"""

import logging
import uuid
from typing import Dict, Optional

from colorama import Fore, Style, init as colorama_init

logger = logging.getLogger(__name__)
colorama_init(autoreset=True)

# Slash-commands understood by the REPL
COMMANDS = {
    "/quit": "Exit Spectra.",
    "/status": "Show current monitoring status and model info.",
    "/train": "Trigger LoRA fine-tuning now.",
    "/search": "Search the web — usage: /search <query>",
    "/scan": "Re-run the initial PC scan.",
    "/memory": "Show recent conversation history.",
    "/help": "List available commands.",
}


class ConversationManager:
    """Manages the interactive REPL and coordinates with all subsystems."""

    def __init__(
        self,
        config: Dict,
        engine,
        database,
        context_builder,
        monitoring_manager=None,
        trainer=None,
        searcher=None,
        scanner=None,
    ) -> None:
        """Initialise the conversation manager.

        Args:
            config: Full parsed config dictionary.
            engine: :class:`spectra.core.engine.AIEngine` instance.
            database: :class:`spectra.memory.database.Database` instance.
            context_builder: :class:`spectra.memory.context.ContextBuilder` instance.
            monitoring_manager: Optional monitoring coordinator.
            trainer: Optional LoRA trainer instance.
            searcher: Optional :class:`spectra.search.web_search.WebSearcher` instance.
            scanner: Optional :class:`spectra.monitoring.scanner.PCScanner` instance.
        """
        self.config = config
        self.engine = engine
        self.db = database
        self.ctx = context_builder
        self.monitoring = monitoring_manager
        self.trainer = trainer
        self.searcher = searcher
        self.scanner = scanner

        self.companion_name: str = config.get("companion", {}).get("name", "Spectra")
        self.training_config = config.get("training", {})
        self.trigger_n: int = self.training_config.get("trigger_every_n_conversations", 50)
        self.training_enabled: bool = self.training_config.get("enabled", True)

        self.session_id: str = str(uuid.uuid4())
        self.turn_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive REPL — blocks until the user quits."""
        print(
            f"\n{Fore.CYAN}{'=' * 60}\n"
            f"  {self.companion_name} is ready. Type {Fore.YELLOW}/help{Fore.CYAN} for commands.\n"
            f"{'=' * 60}{Style.RESET_ALL}\n"
        )

        while True:
            try:
                user_input = input(f"{Fore.GREEN}You > {Style.RESET_ALL}").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_quit = self._handle_command(user_input)
                if should_quit:
                    break
                continue

            self._chat(user_input)

    def inject_proactive_message(self, message: str) -> None:
        """Display a proactive message from Spectra without user input.

        Args:
            message: The AI-generated opener.
        """
        print(
            f"\n{Fore.MAGENTA}[Spectra] {message}{Style.RESET_ALL}\n"
        )
        self.db.save_message("assistant", message, session_id=self.session_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chat(self, user_message: str) -> None:
        """Process one user turn and print Spectra's reply.

        Args:
            user_message: Raw text from the user.
        """
        self.db.save_message("user", user_message, session_id=self.session_id)

        activities = self._get_activities()
        system_prompt = self.ctx.build_system_prompt(current_activities=activities)
        history = self.ctx.build_history(n=10, session_id=self.session_id)
        # Exclude the turn we just saved so it is not duplicated in history
        history = history[:-1] if history and history[-1]["role"] == "user" else history

        print(f"{Fore.YELLOW}{self.companion_name} > {Style.RESET_ALL}", end="", flush=True)
        try:
            response = self.engine.generate_response(
                user_message=user_message,
                context=system_prompt,
                history=history,
                stream=True,
            )
        except Exception as exc:
            logger.error("Generation error: %s", exc)
            response = "Sorry, I ran into an issue. Could you try again?"
            print(response)

        # The streamer already printed the response; add a newline
        print()

        self.db.save_message("assistant", response, session_id=self.session_id)
        self.turn_count += 1
        self._check_auto_train()

    def _get_activities(self) -> Optional[Dict]:
        """Collect live activity data from monitoring modules."""
        if not self.monitoring:
            return None
        try:
            return self.monitoring.get_current_activities()
        except Exception as exc:
            logger.warning("Monitoring error: %s", exc)
            return None

    def _check_auto_train(self) -> None:
        """Prompt the user to train if the conversation threshold is reached."""
        if not self.training_enabled or not self.trainer:
            return
        total = self.db.get_conversation_count()
        if total > 0 and total % self.trigger_n == 0:
            print(
                f"\n{Fore.CYAN}[Spectra] You've had {total} conversation turns. "
                f"Would you like me to fine-tune on them now? (y/n){Style.RESET_ALL} ",
                end="",
            )
            try:
                answer = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer == "y":
                self._run_training()

    def _run_training(self) -> None:
        """Kick off LoRA training and report the result."""
        print(f"{Fore.CYAN}Starting LoRA fine-tuning …{Style.RESET_ALL}")
        try:
            self.trainer.train()
            print(f"{Fore.GREEN}Training complete!{Style.RESET_ALL}")
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            print(f"{Fore.RED}Training failed: {exc}{Style.RESET_ALL}")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_command(self, raw: str) -> bool:
        """Dispatch a slash-command.

        Args:
            raw: Full raw input string including the leading ``/``.

        Returns:
            True if the REPL should exit, False otherwise.
        """
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/quit": lambda _: self._cmd_quit(),
            "/help": lambda _: self._cmd_help(),
            "/status": lambda _: self._cmd_status(),
            "/train": lambda _: self._cmd_train(),
            "/memory": lambda _: self._cmd_memory(),
            "/scan": lambda _: self._cmd_scan(),
            "/search": self._cmd_search,
        }

        handler = handlers.get(cmd)
        if handler is None:
            print(f"{Fore.RED}Unknown command '{cmd}'. Type /help for a list.{Style.RESET_ALL}")
            return False
        return handler(args) or False

    def _cmd_quit(self) -> bool:
        print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
        return True

    def _cmd_help(self) -> None:
        print(f"\n{Fore.CYAN}Available commands:{Style.RESET_ALL}")
        for cmd, desc in COMMANDS.items():
            print(f"  {Fore.YELLOW}{cmd:<20}{Style.RESET_ALL}{desc}")
        print()

    def _cmd_status(self) -> None:
        total = self.db.get_conversation_count()
        activities = self._get_activities() or {}
        print(f"\n{Fore.CYAN}=== Status ==={Style.RESET_ALL}")
        print(f"  Model loaded : {self.engine.is_loaded()}")
        print(f"  Total turns  : {total}")
        print(f"  Session turns: {self.turn_count}")
        print(f"  Session ID   : {self.session_id}")
        if activities:
            for k, v in activities.items():
                if v:
                    print(f"  {k:<13}: {v}")
        print()

    def _cmd_train(self) -> None:
        if not self.trainer:
            print(f"{Fore.RED}Trainer not available.{Style.RESET_ALL}")
            return
        self._run_training()

    def _cmd_memory(self) -> None:
        history = self.db.get_recent_conversations(n=10)
        if not history:
            print(f"{Fore.CYAN}No conversation history yet.{Style.RESET_ALL}")
            return
        print(f"\n{Fore.CYAN}=== Recent conversation history ==={Style.RESET_ALL}")
        for turn in history:
            role_color = Fore.GREEN if turn["role"] == "user" else Fore.YELLOW
            print(f"  {role_color}{turn['role']:<12}{Style.RESET_ALL}{turn['content'][:120]}")
        print()

    def _cmd_scan(self) -> None:
        if not self.scanner:
            print(f"{Fore.RED}Scanner not available.{Style.RESET_ALL}")
            return
        print(f"{Fore.CYAN}Running PC scan …{Style.RESET_ALL}")
        try:
            self.scanner.run()
            print(f"{Fore.GREEN}Scan complete.{Style.RESET_ALL}")
        except Exception as exc:
            logger.error("Scan failed: %s", exc)
            print(f"{Fore.RED}Scan failed: {exc}{Style.RESET_ALL}")

    def _cmd_search(self, query: str) -> None:
        if not query:
            print(f"{Fore.RED}Usage: /search <query>{Style.RESET_ALL}")
            return
        if not self.searcher:
            print(f"{Fore.RED}Web search not available.{Style.RESET_ALL}")
            return
        print(f"{Fore.CYAN}Searching for '{query}' …{Style.RESET_ALL}")
        try:
            results = self.searcher.search(query)
            if not results:
                print("No results found.")
                return
            for i, r in enumerate(results, 1):
                print(f"\n  {Fore.YELLOW}{i}. {r.get('title', 'No title')}{Style.RESET_ALL}")
                print(f"     {r.get('url', '')}")
                print(f"     {r.get('snippet', '')[:200]}")
            print()
        except Exception as exc:
            logger.error("Search error: %s", exc)
            print(f"{Fore.RED}Search failed: {exc}{Style.RESET_ALL}")
