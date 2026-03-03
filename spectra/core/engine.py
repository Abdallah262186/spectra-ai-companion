"""
AI Engine for Spectra.

Handles model loading (Qwen2-1.5B-Instruct with 8-bit quantization),
LoRA adapter loading, and response generation with streaming output.
"""

import logging
import os
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class AIEngine:
    """Manages the Qwen2-1.5B-Instruct model with 8-bit quantization.

    Loads the model once at startup and provides a generate_response()
    method used by the conversation and proactive systems.
    """

    def __init__(self, config: Dict) -> None:
        """Initialise the engine from the model section of config.yaml.

        Args:
            config: The full parsed config dictionary.
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.model_name: str = self.model_config.get("name", "Qwen/Qwen2-1.5B-Instruct")
        self.quantization: str = self.model_config.get("quantization", "8bit")
        self.max_new_tokens: int = self.model_config.get("max_new_tokens", 150)
        self.temperature: float = self.model_config.get("temperature", 0.7)
        self.device: str = self.model_config.get("device", "cuda:0")
        self.adapter_path: str = config.get("training", {}).get(
            "adapter_save_path", "adapters/"
        )

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer into memory.

        Uses 8-bit quantization via bitsandbytes to stay within the
        GTX 1660 Super's 6 GB VRAM budget.  Any saved LoRA adapter is
        applied automatically if found.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info("Loading tokenizer for %s …", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            quantization_config = None
            if self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using 8-bit quantization (bitsandbytes)")

            logger.info("Loading model %s …", self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()

            self._load_adapters()
            self._loaded = True
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    def _load_adapters(self) -> None:
        """Apply the most recent LoRA adapter from the adapters/ directory, if any."""
        if not os.path.isdir(self.adapter_path):
            return

        adapter_dirs = sorted(
            [
                d
                for d in os.listdir(self.adapter_path)
                if os.path.isdir(os.path.join(self.adapter_path, d))
            ]
        )
        if not adapter_dirs:
            return

        latest = os.path.join(self.adapter_path, adapter_dirs[-1])
        try:
            from peft import PeftModel

            logger.info("Loading LoRA adapter from %s …", latest)
            self.model = PeftModel.from_pretrained(self.model, latest)
            self.model = self.model.merge_and_unload()
            logger.info("LoRA adapter applied.")
        except Exception as exc:
            logger.warning("Could not load LoRA adapter: %s", exc)

    def generate_response(
        self,
        user_message: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = True,
    ) -> str:
        """Generate a response for the given user message.

        Builds a ChatML-formatted prompt from the system context and
        conversation history, then runs greedy decoding with optional
        token streaming.

        Args:
            user_message: The latest message from the user.
            context: System prompt text assembled by the context builder.
            history: List of previous turns as {"role": …, "content": …}.
            stream: Whether to print tokens as they are generated.

        Returns:
            The assistant's response as a plain string.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages: List[Dict[str, str]] = []
        if context:
            messages.append({"role": "system", "content": context})
        for turn in (history or []):
            messages.append(turn)
        messages.append({"role": "user", "content": user_message})

        # Apply the Qwen2 ChatML template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        response_text = self._generate(input_ids, stream=stream)
        return response_text

    def _generate(self, input_ids: "torch.Tensor", stream: bool = True) -> str:
        """Run the forward pass and decode tokens.

        Falls back to a shorter generation window if CUDA runs out of memory.

        Args:
            input_ids: Tokenised input tensor already on the model device.
            stream: Print each token to stdout as it is generated.

        Returns:
            Decoded response string.
        """
        max_new_tokens = self.max_new_tokens
        for attempt in range(2):
            try:
                with torch.no_grad():
                    if stream:
                        return self._stream_generate(input_ids, max_new_tokens)
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        do_sample=self.temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                new_tokens = output[0][input_ids.shape[-1]:]
                return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError:
                if attempt == 0:
                    logger.warning("CUDA OOM — retrying with reduced max_new_tokens.")
                    torch.cuda.empty_cache()
                    max_new_tokens = max(32, max_new_tokens // 2)
                else:
                    logger.error("CUDA OOM on second attempt — returning fallback message.")
                    torch.cuda.empty_cache()
                    return "I'm a bit overwhelmed right now — could you try again?"

    def _stream_generate(self, input_ids: "torch.Tensor", max_new_tokens: int) -> str:
        """Generate and print tokens one at a time for a real-time feel.

        Args:
            input_ids: Tokenised input tensor already on the model device.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Full decoded response string.
        """
        from transformers import TextStreamer

        streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        new_tokens = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def is_loaded(self) -> bool:
        """Return True if the model has been successfully loaded."""
        return self._loaded
