"""
Semantic Embedding Module using PubMedBERT

This module provides semantic embeddings for body parts and organs
using pre-trained medical domain language models.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
import logging

# Lazy import to avoid loading transformers if not needed
_transformers_available = None

def _check_transformers():
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    return _transformers_available


class SemanticEmbeddingCache:
    """
    Cache for pre-computed semantic embeddings.
    Avoids re-computing embeddings during training.
    """
    _instance = None
    _cache: Dict[str, torch.Tensor] = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._cache.get(key)
    
    def set(self, key: str, value: torch.Tensor):
        self._cache[key] = value
    
    def clear(self):
        self._cache.clear()


class BodyPartEmbedding(nn.Module):
    """
    Body Part Embedding using PubMedBERT semantic features.
    
    Provides:
    - Semantic embedding from pre-trained medical LM
    - Optional learnable refinement
    - Time conditioning MLP for diffusion
    - Channel conditioning (spatial broadcast)
    
    Args:
        body_parts: List of body part names
        output_dim: Output embedding dimension
        time_embed_dim: Dimension for time conditioning (UNet time_embed)
        use_semantic: Whether to use semantic embeddings (requires transformers)
        use_learnable: Whether to add learnable embeddings
        freeze_semantic: Whether to freeze semantic encoder
    """
    
    def __init__(
        self,
        body_parts: List[str],
        output_dim: int = 64,
        time_embed_dim: int = 512,
        use_semantic: bool = True,
        use_learnable: bool = True,
        freeze_semantic: bool = True,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        prompt_template: str = "MRI axial slice of the {name} region",
    ):
        super().__init__()
        
        self.num_body_parts = len(body_parts)
        self.body_parts = body_parts
        self.output_dim = output_dim
        self.use_semantic = use_semantic and _check_transformers()
        self.use_learnable = use_learnable
        
        # Dimension allocation
        if self.use_semantic and self.use_learnable:
            self.semantic_dim = output_dim // 2
            self.learnable_dim = output_dim - self.semantic_dim
        elif self.use_semantic:
            self.semantic_dim = output_dim
            self.learnable_dim = 0
        else:
            self.semantic_dim = 0
            self.learnable_dim = output_dim
        
        # ===== Semantic Embedding (from PubMedBERT) =====
        if self.use_semantic:
            self._init_semantic_embedding(
                body_parts, model_name, prompt_template, freeze_semantic
            )
        
        # ===== Learnable Embedding =====
        if self.use_learnable:
            self.learnable_embedding = nn.Embedding(self.num_body_parts, self.learnable_dim)
            nn.init.normal_(self.learnable_embedding.weight, std=0.02)
        
        # ===== Time Conditioning MLP =====
        self.time_mlp = nn.Sequential(
            nn.Linear(output_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # Zero initialization for stable training
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        
        logging.info(
            f"[BodyPartEmbedding] Initialized with {self.num_body_parts} body parts, "
            f"output_dim={output_dim}, semantic={self.use_semantic}, learnable={self.use_learnable}"
        )
    
    def _init_semantic_embedding(
        self,
        body_parts: List[str],
        model_name: str,
        prompt_template: str,
        freeze: bool,
    ):
        """Initialize semantic embeddings from PubMedBERT."""
        from transformers import AutoTokenizer, AutoModel
        
        logging.info(f"[BodyPartEmbedding] Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        
        if freeze:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        
        # Create prompts and encode
        prompts = [prompt_template.format(name=bp) for bp in body_parts]
        
        with torch.no_grad():
            inputs = tokenizer(
                prompts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=64,
            )
            outputs = encoder(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]  # [num_body_parts, 768]
        
        # Register as buffer (not trainable, but saved with model)
        self.register_buffer("semantic_embeddings", embeddings)
        
        # Projection layer
        self.semantic_proj = nn.Linear(embeddings.shape[-1], self.semantic_dim)
        
        # Clean up encoder to save memory
        del encoder, tokenizer
        
        logging.info(
            f"[BodyPartEmbedding] Semantic embeddings computed: {embeddings.shape}"
        )
    
    def forward(self, body_part_id: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for body part IDs.
        
        Args:
            body_part_id: [B] tensor of body part indices
            
        Returns:
            embedding: [B, output_dim] body part embedding
        """
        parts = []
        
        # Semantic part
        if self.use_semantic:
            semantic = self.semantic_embeddings[body_part_id]  # [B, 768]
            semantic = self.semantic_proj(semantic)  # [B, semantic_dim]
            parts.append(semantic)
        
        # Learnable part
        if self.use_learnable:
            learnable = self.learnable_embedding(body_part_id)  # [B, learnable_dim]
            parts.append(learnable)
        
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)  # [B, output_dim]
    
    def get_time_embedding(self, body_part_id: torch.Tensor) -> torch.Tensor:
        """Get time conditioning embedding."""
        embed = self.forward(body_part_id)
        return self.time_mlp(embed)
    
    def get_spatial_embedding(
        self, 
        body_part_id: torch.Tensor, 
        height: int, 
        width: int,
    ) -> torch.Tensor:
        """
        Get spatially broadcast embedding for channel conditioning.
        
        Args:
            body_part_id: [B] tensor
            height, width: Target spatial dimensions
            
        Returns:
            spatial_embed: [B, output_dim, H, W]
        """
        embed = self.forward(body_part_id)  # [B, output_dim]
        spatial = embed.unsqueeze(-1).unsqueeze(-1)  # [B, output_dim, 1, 1]
        return spatial.expand(-1, -1, height, width)  # [B, output_dim, H, W]


class OrganEmbedding(nn.Module):
    """
    Organ Embedding using PubMedBERT semantic features.
    
    Handles multi-label organ encoding with aggregation.
    
    Args:
        organs: List of organ names
        output_dim: Output embedding dimension
        use_semantic: Whether to use semantic embeddings
        use_learnable: Whether to add learnable embeddings
        aggregation: How to aggregate multiple organs ("mean", "attention", "weighted")
    """
    
    def __init__(
        self,
        organs: List[str],
        output_dim: int = 64,
        use_semantic: bool = True,
        use_learnable: bool = True,
        freeze_semantic: bool = True,
        aggregation: str = "weighted",
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        prompt_template: str = "MRI showing the {name} anatomical structure",
    ):
        super().__init__()
        
        self.num_organs = len(organs)
        self.organs = organs
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.use_semantic = use_semantic and _check_transformers()
        self.use_learnable = use_learnable
        
        # Dimension allocation
        if self.use_semantic and self.use_learnable:
            self.semantic_dim = output_dim // 2
            self.learnable_dim = output_dim - self.semantic_dim
        elif self.use_semantic:
            self.semantic_dim = output_dim
            self.learnable_dim = 0
        else:
            self.semantic_dim = 0
            self.learnable_dim = output_dim
        
        # ===== Semantic Embedding (from PubMedBERT) =====
        if self.use_semantic:
            self._init_semantic_embedding(
                organs, model_name, prompt_template, freeze_semantic
            )
        
        # ===== Learnable Embedding =====
        if self.use_learnable:
            self.learnable_embedding = nn.Embedding(self.num_organs, self.learnable_dim)
            nn.init.normal_(self.learnable_embedding.weight, std=0.02)
        
        # ===== Aggregation =====
        if aggregation == "weighted":
            # Learnable importance weights per organ
            self.organ_importance = nn.Parameter(torch.zeros(self.num_organs))
        elif aggregation == "attention":
            # Self-attention for aggregation
            self.attn_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
            self.attn_proj = nn.Linear(output_dim, output_dim)
        
        logging.info(
            f"[OrganEmbedding] Initialized with {self.num_organs} organs, "
            f"output_dim={output_dim}, aggregation={aggregation}"
        )
    
    def _init_semantic_embedding(
        self,
        organs: List[str],
        model_name: str,
        prompt_template: str,
        freeze: bool,
    ):
        """Initialize semantic embeddings from PubMedBERT."""
        from transformers import AutoTokenizer, AutoModel
        
        logging.info(f"[OrganEmbedding] Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        
        if freeze:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        
        # Create prompts and encode
        prompts = [prompt_template.format(name=organ) for organ in organs]
        
        with torch.no_grad():
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=64,
            )
            outputs = encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [num_organs, 768]
        
        self.register_buffer("semantic_embeddings", embeddings)
        self.semantic_proj = nn.Linear(embeddings.shape[-1], self.semantic_dim)
        
        del encoder, tokenizer
        
        logging.info(f"[OrganEmbedding] Semantic embeddings computed: {embeddings.shape}")
    
    def _get_all_embeddings(self) -> torch.Tensor:
        """Get combined embeddings for all organs."""
        parts = []
        
        if self.use_semantic:
            semantic = self.semantic_proj(self.semantic_embeddings)  # [num_organs, semantic_dim]
            parts.append(semantic)
        
        if self.use_learnable:
            learnable = self.learnable_embedding.weight  # [num_organs, learnable_dim]
            parts.append(learnable)
        
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)  # [num_organs, output_dim]
    
    def forward(self, organ_mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate organ embeddings based on presence mask.
        
        Args:
            organ_mask: [B, num_organs] multi-hot tensor
            
        Returns:
            aggregated: [B, output_dim] aggregated organ embedding
        """
        B = organ_mask.shape[0]
        all_embeds = self._get_all_embeddings()  # [num_organs, output_dim]
        all_embeds = all_embeds.unsqueeze(0).expand(B, -1, -1)  # [B, num_organs, output_dim]
        
        # Apply mask
        mask = organ_mask.unsqueeze(-1).float()  # [B, num_organs, 1]
        
        if self.aggregation == "mean":
            # Simple mean of present organs
            weighted = all_embeds * mask
            count = mask.sum(dim=1).clamp(min=1)
            aggregated = weighted.sum(dim=1) / count
            
        elif self.aggregation == "weighted":
            # Importance-weighted mean
            importance = torch.softmax(self.organ_importance, dim=0)  # [num_organs]
            importance = importance.unsqueeze(0).unsqueeze(-1)  # [1, num_organs, 1]
            weighted_mask = mask * importance
            weighted = all_embeds * weighted_mask
            norm = weighted_mask.sum(dim=1).clamp(min=1e-6)
            aggregated = weighted.sum(dim=1) / norm
            
        elif self.aggregation == "attention":
            # Attention-based aggregation
            query = self.attn_query.expand(B, -1, -1)  # [B, 1, output_dim]
            
            # Compute attention scores
            scores = torch.bmm(query, all_embeds.transpose(1, 2))  # [B, 1, num_organs]
            scores = scores.squeeze(1)  # [B, num_organs]
            
            # Mask out absent organs
            scores = scores.masked_fill(~organ_mask.bool(), float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)  # [B, num_organs]
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            
            # Weighted sum
            aggregated = torch.bmm(
                attn_weights.unsqueeze(1),  # [B, 1, num_organs]
                all_embeds  # [B, num_organs, output_dim]
            ).squeeze(1)  # [B, output_dim]
            
            aggregated = self.attn_proj(aggregated)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return aggregated
    
    def get_all_embeddings_for_crossattn(self) -> torch.Tensor:
        """
        Get all organ embeddings for cross-attention (Key/Value).

        Returns:
            embeddings: [num_organs, output_dim]
        """
        return self._get_all_embeddings()

    def get_expanded_embeddings(self, organ_mask: torch.Tensor) -> tuple:
        """
        Get per-organ embeddings expanded for cross-attention, with key_padding_mask.

        Args:
            organ_mask: [B, num_organs] multi-hot tensor

        Returns:
            organ_kv: [B, num_organs, output_dim] all organ embeddings (broadcast)
            key_padding_mask: [B, num_organs] bool, True for absent organs (to be masked)
        """
        B = organ_mask.shape[0]
        all_embeds = self._get_all_embeddings()  # [num_organs, output_dim]
        organ_kv = all_embeds.unsqueeze(0).expand(B, -1, -1)  # [B, num_organs, output_dim]
        # nn.MultiheadAttention key_padding_mask: True = ignore
        key_padding_mask = ~organ_mask.bool()  # [B, num_organs]
        return organ_kv, key_padding_mask


class FallbackEmbedding(nn.Module):
    """
    Fallback embedding when transformers is not available.
    Uses pure learnable embeddings.
    """
    
    def __init__(
        self,
        num_classes: int,
        output_dim: int = 64,
        time_embed_dim: int = 512,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_classes, output_dim)
        self.output_dim = output_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(output_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(idx)
    
    def get_time_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.forward(idx))
    
    def get_spatial_embedding(
        self, idx: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        embed = self.forward(idx)
        spatial = embed.unsqueeze(-1).unsqueeze(-1)
        return spatial.expand(-1, -1, height, width)


def create_body_part_embedding(
    body_parts: List[str],
    output_dim: int = 64,
    time_embed_dim: int = 512,
    use_semantic: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create body part embedding module.
    Falls back to learnable embedding if transformers unavailable.
    """
    if use_semantic and _check_transformers():
        return BodyPartEmbedding(
            body_parts=body_parts,
            output_dim=output_dim,
            time_embed_dim=time_embed_dim,
            use_semantic=True,
            **kwargs,
        )
    else:
        logging.warning(
            "[create_body_part_embedding] transformers not available, "
            "using fallback learnable embedding"
        )
        return FallbackEmbedding(
            num_classes=len(body_parts),
            output_dim=output_dim,
            time_embed_dim=time_embed_dim,
        )


def create_organ_embedding(
    organs: List[str],
    output_dim: int = 64,
    use_semantic: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create organ embedding module.
    """
    if use_semantic and _check_transformers():
        return OrganEmbedding(
            organs=organs,
            output_dim=output_dim,
            use_semantic=True,
            **kwargs,
        )
    else:
        logging.warning(
            "[create_organ_embedding] transformers not available, "
            "using fallback"
        )
        # For organs, we need a multi-label handler
        return OrganEmbedding(
            organs=organs,
            output_dim=output_dim,
            use_semantic=False,
            use_learnable=True,
            **kwargs,
        )

