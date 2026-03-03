"""
LoRA / QLoRA fine-tuner for Spectra.

Loads the base Qwen2-1.5B-Instruct model in 8-bit, attaches a LoRA
adapter, trains on the JSONL file produced by data_prep.py, and saves
the adapter to the adapters/ directory.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Wraps PEFT + TRL to fine-tune Spectra's base model with LoRA."""

    def __init__(self, config: Dict, database) -> None:
        """Initialise the trainer from the training section of config.yaml.

        Args:
            config: Full parsed config dictionary.
            database: :class:`spectra.memory.database.Database` instance.
        """
        self.config = config
        self.db = database
        training_cfg = config.get("training", {})
        self.lora_rank: int = training_cfg.get("lora_rank", 8)
        self.lora_alpha: int = training_cfg.get("lora_alpha", 16)
        self.learning_rate: float = training_cfg.get("learning_rate", 2e-4)
        self.epochs: int = training_cfg.get("epochs", 3)
        self.adapter_save_path: str = training_cfg.get("adapter_save_path", "adapters/")
        self.model_name: str = config.get("model", {}).get("name", "Qwen/Qwen2-1.5B-Instruct")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> str:
        """Run the full training pipeline.

        Steps:
            1. Prepare JSONL data from conversation history.
            2. Load base model in 8-bit.
            3. Attach LoRA adapter.
            4. Train with SFTTrainer (or fallback manual loop).
            5. Save adapter and log to database.

        Returns:
            Path where the adapter was saved.
        """
        from spectra.training.data_prep import prepare_training_data

        logger.info("Preparing training data …")
        last_id = self.db.get_last_training_id()
        jsonl_path, num_examples = prepare_training_data(self.db, since_conversation_count=last_id)

        if num_examples == 0:
            logger.warning("No training examples found — skipping training.")
            return ""

        logger.info("Loading model for training …")
        model, tokenizer = self._load_model()

        logger.info("Attaching LoRA adapter …")
        model = self._attach_lora(model)

        adapter_path = self._make_adapter_path()
        start = time.time()

        logger.info("Starting training on %d examples …", num_examples)
        try:
            self._train_sft(model, tokenizer, jsonl_path, adapter_path)
        except ImportError:
            logger.warning("TRL not available — falling back to manual training loop.")
            self._train_manual(model, tokenizer, jsonl_path, adapter_path)
        except Exception as exc:
            logger.error("Training error: %s", exc)
            raise

        duration = time.time() - start
        self.db.save_training_log(num_examples, duration, adapter_path)
        logger.info("Training complete in %.1f s. Adapter saved to %s", duration, adapter_path)

        # Clean up the temp JSONL file
        try:
            os.remove(jsonl_path)
        except OSError:
            pass

        return adapter_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load Qwen2 in 8-bit for QLoRA training.

        Returns:
            Tuple of (model, tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, tokenizer

    def _attach_lora(self, model):
        """Add LoRA parameters to the model.

        Args:
            model: HuggingFace causal LM model.

        Returns:
            PEFT model with LoRA applied.
        """
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def _make_adapter_path(self) -> str:
        """Generate a timestamped output directory for the adapter.

        Returns:
            Absolute path string for the new adapter directory.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.adapter_save_path, f"adapter_{ts}")
        os.makedirs(path, exist_ok=True)
        return path

    def _train_sft(self, model, tokenizer, jsonl_path: str, adapter_path: str) -> None:
        """Train using TRL's SFTTrainer.

        Args:
            model: PEFT model with LoRA.
            tokenizer: Corresponding tokenizer.
            jsonl_path: Path to the JSONL training file.
            adapter_path: Directory to save the trained adapter.
        """
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig

        dataset = load_dataset("json", data_files=jsonl_path, split="train")

        sft_config = SFTConfig(
            output_dir=adapter_path,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=1,
            dataset_text_field="text",
            max_seq_length=512,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=sft_config,
            tokenizer=tokenizer,
        )
        trainer.train()
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

    def _train_manual(self, model, tokenizer, jsonl_path: str, adapter_path: str) -> None:
        """Minimal manual training loop as a fallback when TRL is absent.

        Args:
            model: PEFT model with LoRA.
            tokenizer: Corresponding tokenizer.
            jsonl_path: Path to the JSONL training file.
            adapter_path: Directory to save the trained adapter.
        """
        import json

        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset

        class _TextDataset(Dataset):
            def __init__(self, path, tok, max_len):
                self.samples = []
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line.strip())
                        enc = tok(
                            item["text"],
                            truncation=True,
                            max_length=max_len,
                            return_tensors="pt",
                        )
                        self.samples.append(enc["input_ids"].squeeze(0))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        dataset = _TextDataset(jsonl_path, tokenizer, max_len=512)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for step, batch in enumerate(loader):
                batch = batch.to(model.device)
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                if step % 10 == 0:
                    logger.info("Epoch %d step %d loss=%.4f", epoch + 1, step, loss.item())
            logger.info("Epoch %d avg loss: %.4f", epoch + 1, total_loss / max(len(loader), 1))

        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
