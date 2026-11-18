"""Turn text into fixed-length sentence embeddings.

How it works:
    1. A Hugging Face tokenizer splits each text into tokens.
    2. A transformer model (MiniLM by default) produces a vector per token.
    3. Token vectors are averaged, ignoring padding tokens (mean pooling).
    4. Each averaged vector is scaled to length 1 (L2 normalisation).

Only Hugging Face Transformers and PyTorch are needed.
The sentence-transformers package is not required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextEmbedder:
    """Convert any list of texts into unit-length embedding vectors.

    The default model returns 384-dimensional vectors.
    """

    CONFIG_FILE_NAME = "embedder_config.json"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 256,
        device: str | None = None,
        local_files_only: bool = False,
    ):
        """Load the tokenizer and model.

        Args:
            model_name: Hugging Face model id, or a local folder path.
            max_tokens: Longest token sequence per text; longer texts are cut off.
            device: "cuda" or "cpu". Picks the GPU automatically when available.
            local_files_only: When True, never download; load from disk only.
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be a positive number.")

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only, trust_remote_code=False
        )
        self.model.to(self.device).eval()

    @property
    def embedding_size(self) -> int:
        """Number of values in each embedding vector."""
        return self.model.config.hidden_size

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 64,
        show_progress: bool = False,
        as_numpy: bool = True,
    ):
        """Return one embedding row per input text.

        Args:
            texts: A single string or a list of strings.
            batch_size: How many texts to process at once.
            show_progress: Print a progress line every 100 batches and at the end.
            as_numpy: Return a NumPy array when True, a PyTorch tensor when False.

        Returns:
            A float32 array of shape (number of texts, embedding_size).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be a positive number.")
        if isinstance(texts, str):
            texts = [texts]

        batches = []
        total = len(texts)

        with torch.inference_mode():
            for start in range(0, total, batch_size):
                batch_texts = texts[start:start + batch_size]
                batch_vectors = self._encode_batch(batch_texts)
                batches.append(batch_vectors)

                if show_progress:
                    self._print_progress(start, batch_size, total)

        if batches:
            embeddings = np.concatenate(batches).astype(np.float32, copy=False)
        else:
            embeddings = np.empty((0, self.embedding_size), dtype=np.float32)

        return embeddings if as_numpy else torch.from_numpy(embeddings)

    def _encode_batch(self, batch_texts: list[str]) -> np.ndarray:
        """Tokenize one batch, run the model, mean-pool, and normalise."""
        tokens = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        tokens = {name: values.to(self.device) for name, values in tokens.items()}

        # One vector per token: shape (batch, tokens, embedding_size).
        token_vectors = self.model(**tokens).last_hidden_state

        # 1 for real tokens, 0 for padding, so padding does not affect the average.
        mask = tokens["attention_mask"].unsqueeze(-1).to(token_vectors.dtype)
        summed = (token_vectors * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_vectors = summed / token_counts

        # Scale every vector to length 1 so dot product equals cosine similarity.
        unit_vectors = F.normalize(mean_vectors, p=2, dim=1)
        return unit_vectors.cpu().float().numpy()

    @staticmethod
    def _print_progress(start: int, batch_size: int, total: int) -> None:
        """Print progress every 100 batches and after the last batch."""
        done = min(start + batch_size, total)
        batch_number = start // batch_size + 1
        if done >= total or batch_number % 100 == 0:
            print(f"Encoded {done:,}/{total:,} texts")

    def save(self, folder: str | Path) -> None:
        """Save the tokenizer, model weights, and settings to a folder."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_pretrained(str(folder))
        self.model.save_pretrained(str(folder), safe_serialization=True)

        settings = {"max_tokens": self.max_tokens}
        (folder / self.CONFIG_FILE_NAME).write_text(json.dumps(settings) + "\n")

    @classmethod
    def load(cls, folder: str | Path, device: str | None = None) -> "TextEmbedder":
        """Load a saved embedder from a folder without going online."""
        folder = Path(folder)
        settings = json.loads((folder / cls.CONFIG_FILE_NAME).read_text())
        return cls(
            model_name=str(folder),
            max_tokens=settings["max_tokens"],
            device=device,
            local_files_only=True,
        )
