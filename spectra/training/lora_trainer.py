"""LoRA / QLoRA fine-tuning trainer.

Trains a LoRA adapter on top of the base Qwen2-1.5B-Instruct model (loaded in
8-bit) using conversation history.  Uses ``trl.SFTTrainer`` when available,
falling back to a simple manual training loop otherwise.

The adapter is saved with a timestamp so multiple checkpoints are preserved.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Orchestrates QLoRA fine-tuning on Spectra's conversation history.

    Args:
        config: Parsed configuration dictionary.
        db: Initialised :class:`~spectra.memory.database.Database` instance.
    """

    def __init__(self, config: Dict[str, Any], db: Any) -> None:
        self.config = config
        self.db = db
        training_cfg = config.get("training", {})
        self.lora_rank: int = training_cfg.get("lora_rank", 8)
        self.lora_alpha: int = training_cfg.get("lora_alpha", 16)
        self.learning_rate: float = training_cfg.get("learning_rate", 0.0002)
        self.epochs: int = training_cfg.get("epochs", 3)
        self.adapter_save_path: str = training_cfg.get("adapter_save_path", "adapters/")
        model_name: str = config.get("model", {}).get("name", "Qwen/Qwen2-1.5B-Instruct")
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> Optional[str]:
        """Prepare data and run the LoRA training loop.

        Returns:
            Path to the saved adapter, or None if training was skipped.
        """
        from spectra.training.data_prep import DataPrep  # noqa: PLC0415

        prep = DataPrep(self.config, self.db)
        n_samples, data_path, last_id = prep.prepare()

        if n_samples == 0:
            logger.info("Nothing to train on – skipping.")
            return None

        logger.info("Starting LoRA training on %d samples.", n_samples)
        start_time = time.time()

        # Timestamp-stamped output directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        adapter_out = str(Path(self.adapter_save_path) / f"adapter_{ts}")

        try:
            self._run_training(data_path, adapter_out)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("CUDA OOM during training – try reducing batch size.")
                return None
            raise

        duration = time.time() - start_time
        self.db.log_training(
            conversations_used=n_samples,
            duration=round(duration, 1),
            adapter_path=adapter_out,
            last_conversation_id=last_id,
        )
        logger.info("Training complete in %.1f s. Adapter saved to %s", duration, adapter_out)
        return adapter_out

    # ------------------------------------------------------------------
    # Training implementation
    # ------------------------------------------------------------------

    def _run_training(self, data_path: str, adapter_out: str) -> None:
        """Execute the actual training via SFTTrainer or a manual loop.

        Args:
            data_path: Path to the JSONL training file.
            adapter_out: Directory to save the LoRA adapter.
        """
        try:
            self._train_with_sft(data_path, adapter_out)
        except ImportError:
            logger.info("trl not available – falling back to manual training loop.")
            self._train_manual(data_path, adapter_out)

    def _build_peft_model(self) -> tuple:  # type: ignore[type-arg]
        """Load the base model in 8-bit and wrap it with a LoRA config.

        Returns:
            ``(model, tokenizer)`` tuple ready for fine-tuning.
        """
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
        from peft import LoraConfig, get_peft_model, TaskType  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer

    def _train_with_sft(self, data_path: str, adapter_out: str) -> None:
        """Train using ``trl.SFTTrainer``.

        Args:
            data_path: Path to the JSONL training file.
            adapter_out: Directory to save the adapter.

        Raises:
            ImportError: If ``trl`` is not installed.
        """
        from trl import SFTTrainer, SFTConfig  # noqa: PLC0415
        from datasets import load_dataset  # noqa: PLC0415

        model, tokenizer = self._build_peft_model()
        dataset = load_dataset("json", data_files={"train": data_path}, split="train")

        sft_config = SFTConfig(
            output_dir=adapter_out,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            save_steps=50,
            logging_steps=10,
            fp16=False,
            bf16=False,
            optim="adamw_torch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
        )
        trainer.train()
        model.save_pretrained(adapter_out)
        tokenizer.save_pretrained(adapter_out)
        logger.info("SFTTrainer saved adapter to %s", adapter_out)

    def _train_manual(self, data_path: str, adapter_out: str) -> None:
        """Minimal manual training loop (fallback when trl is unavailable).

        Args:
            data_path: Path to the JSONL training file.
            adapter_out: Directory to save the adapter.
        """
        import json  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from torch.optim import AdamW  # noqa: PLC0415

        model, tokenizer = self._build_peft_model()
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        with open(data_path, "r", encoding="utf-8") as fh:
            samples = [json.loads(line) for line in fh if line.strip()]

        model.train()
        device = next(model.parameters()).device

        for epoch in range(self.epochs):
            total_loss = 0.0
            for sample in samples:
                messages = sample.get("messages", [])
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                labels = enc["input_ids"].clone()
                outputs = model(**enc, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(samples), 1)
            logger.info("Epoch %d/%d – avg loss: %.4f", epoch + 1, self.epochs, avg_loss)

        model.save_pretrained(adapter_out)
        tokenizer.save_pretrained(adapter_out)
        logger.info("Manual trainer saved adapter to %s", adapter_out)
