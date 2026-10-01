"""
Text Embedding Pipeline using frozen CLIP text encoder.

This module loads a pretrained HuggingFace CLIP text model and converts natural language
change queries (e.g., "new industrial buildings") into embeddings compatible with the image
element space. The text tower is frozen; only the adapter's parameters are trained.
"""
from src import _cache  # sets HF_HOME before transformers import
import torch
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModel

# the English-only OpenAI CLIP ViT-L/14 text tower (not remote-sensing specific)
text_encoder_name = "openai/clip-vit-large-patch14"


class FrozenTextEncoder:
    """
    Wrapper around CLIP's frozen text encoder.

    This class loads the tokenizer and transformer, caches embeddings,
    and provides a convenient API for converting natural language queries
    into multimodal-compatible embeddings.
    """

    def __init__(
        self,
        model_name: str = text_encoder_name,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the frozen text encoder.

        Args:
            model_name (str): HuggingFace model identifier. Default is CLIP ViT-L/14.
            device (torch.device, optional): Device to run inference on ('cuda' or 'cpu').
                Auto-detected if None. Falls back to CPU if CUDA not available.
            cache_dir (str, optional): Directory for caching downloaded models.
        """
        self.model_name = model_name
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.cache_dir = cache_dir or _cache.CLIP_CACHE_DIR

        # Load tokenizer and model
        print(f"Loading text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        ).to(self.device)

        # Freeze all parameters (critical for low-compute training)
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Text encoder loaded on {self.device}")

    def encode(
        self,
        texts: Union[str, list]
    ) -> torch.Tensor:
        """
        Convert natural language queries into embeddings.

        Args:
            texts (str or list): Single query string or list of query strings.
                Examples: "new industrial buildings", "coastal erosion after storm"

        Returns:
            torch.Tensor: Embeddings in the multimodal shared space.
                Shape: [num_texts, embed_dim]
                For CLIP ViT-L/14: [N, 768]

        Example:
            >>> encoder = FrozenTextEncoder()
            >>> query = "construction on agricultural land"
            >>> emb = encoder.encode(query)
            >>> print(emb.shape)  # torch.Size([1, 768])
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize with padding/truncation for batch processing
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP's default max sequence length
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_out = self.model.text_model(**encoded)
            embeddings = self.model.text_projection(text_out.pooler_output)

        return embeddings

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode multiple queries with automatic batching.

        Useful when encoding large datasets of negative descriptions or validation queries.

        Args:
            texts (list): List of query strings.
            batch_size (int): Batch size for memory efficiency. Default 32.

        Returns:
            torch.Tensor: Concatenated embeddings from all batches.
        """
        all_embeddings = []
        total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            emb = self.encode(batch_texts)
            all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)

    def __len__(self) -> int:
        """Return the embedding dimension."""
        # CLIP uses projection_dim for the multimodal shared space dimension
        return self.model.config.projection_dim
