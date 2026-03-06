"""AI Engine – model loading, inference, and token streaming.

Loads Qwen2-1.5B-Instruct with 8-bit quantisation via bitsandbytes so the
model occupies ≤ 2 GB VRAM on the GTX 1660 Super.  LoRA adapters are applied
automatically if they exist in the configured adapter directory.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class AIEngine:
    """Wraps HuggingFace model loading and text generation.

    Args:
        config: Parsed configuration dictionary (from config.yaml).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_config = config.get("model", {})
        self.model_name: str = self.model_config.get("name", "Qwen/Qwen2-1.5B-Instruct")
        self.quantization: str = self.model_config.get("quantization", "8bit")
        self.max_new_tokens: int = self.model_config.get("max_new_tokens", 150)
        self.temperature: float = self.model_config.get("temperature", 0.7)
        self.device: str = self.model_config.get("device", "cuda:0")
        self.adapter_path: str = config.get("training", {}).get("adapter_save_path", "adapters/")

        self.model = None
        self.tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load tokenizer and model with 8-bit quantisation.

        Applies any existing LoRA adapter found in *adapter_save_path*.
        Falls back to CPU if CUDA is unavailable.
        """
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415

        logger.info("Loading tokenizer: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logger.warning("CUDA not available – loading model on CPU (slow).")

        quant_config = None
        if self.quantization == "8bit" and use_cuda:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info("Loading model: %s (quantisation=%s)", self.model_name, self.quantization)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map=self.device if use_cuda else "cpu",
            trust_remote_code=True,
        )

        # Apply LoRA adapter if one exists
        self._try_load_adapter()

        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully.")

    def _try_load_adapter(self) -> None:
        """Apply the most recent LoRA adapter if the adapter directory exists."""
        try:
            from peft import PeftModel  # noqa: PLC0415
        except ImportError:
            logger.debug("peft not installed; skipping adapter loading.")
            return

        adapter_dir = Path(self.adapter_path)
        if not adapter_dir.exists():
            return

        # Pick the most recently modified adapter sub-directory
        sub_dirs = sorted(
            [d for d in adapter_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not sub_dirs:
            return

        latest = sub_dirs[0]
        logger.info("Loading LoRA adapter from %s", latest)
        try:
            self.model = PeftModel.from_pretrained(self.model, str(latest))
            logger.info("LoRA adapter applied.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load adapter: %s", exc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
    ) -> str:
        """Generate a response given a list of chat messages.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            stream: If True, print tokens to stdout as they are generated.

        Returns:
            The complete assistant response text.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch  # noqa: PLC0415

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        # Move input tensors to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated = ""
        try:
            if stream:
                generated = self._stream_generate(inputs)
            else:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                generated = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning("CUDA OOM – retrying with reduced max_new_tokens.")
                import gc  # noqa: PLC0415
                gc.collect()
                torch.cuda.empty_cache()
                return self._fallback_generate(inputs)
            raise

        return generated.strip()

    def _stream_generate(self, inputs: Dict[str, Any]) -> str:
        """Generate tokens one-by-one and print them as they arrive.

        Args:
            inputs: Tokenised model inputs.

        Returns:
            The complete decoded response string.
        """
        import torch  # noqa: PLC0415
        from transformers import TextIteratorStreamer  # noqa: PLC0415
        import threading  # noqa: PLC0415

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        tokens: List[str] = []
        for token in streamer:
            print(token, end="", flush=True)
            tokens.append(token)

        thread.join()
        return "".join(tokens)

    def _fallback_generate(self, inputs: Dict[str, Any]) -> str:
        """Retry generation with a much smaller token budget after OOM.

        Args:
            inputs: Tokenised model inputs.

        Returns:
            The decoded response string (possibly truncated).
        """
        import torch  # noqa: PLC0415

        reduced = max(32, self.max_new_tokens // 3)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=reduced,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
