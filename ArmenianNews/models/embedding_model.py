import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Tuple
import gc


class EmbeddingModel:
    def __init__(self, model_name: str = "Metric-AI/armenian-text-embeddings-1", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        # Загрузка модели и токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        ).to(self.device)

        # Включаем gradient checkpointing для экономии памяти
        self.model.gradient_checkpointing_enable()

    @staticmethod
    def mean_pooling(model_output: Tuple[Tensor, ...], attention_mask: Tensor) -> Tensor:
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embeddings_batch(self, texts: list, batch_size: int = 16) -> Tensor:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_dict = self.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch_dict)

            batch_embeddings = self.average_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def get_embeddings_single(self, text: str) -> Tensor:
        return self.get_embeddings_batch([text])[0]


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()