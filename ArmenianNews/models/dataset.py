import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List


class ArmenianNewsDataset(Dataset):
    """Датасет для новостей на армянском языке"""

    def __init__(self, texts: List[str], dates: List, titles: List[str]):
        self.texts = texts
        self.dates = dates
        self.titles = titles

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        return {
            'text': self.texts[idx],
            'date': self.dates[idx],
            'title': self.titles[idx],
            'index': idx
        }


class OptimizedTripletDataset(Dataset):
    """Оптимизированный датасет для триплетов"""

    def __init__(self, texts: List[str], triplets: List[Dict],
                 tokenizer: AutoTokenizer, max_length: int = 128):
        self.texts = texts
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict:
        triplet = self.triplets[idx]

        anchor_text = self.texts[triplet['anchor']]
        positive_text = self.texts[triplet['positive']]
        negative_text = self.texts[triplet['negative']]

        # Токенизация с оптимизацией
        anchor_enc = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        positive_enc = self.tokenizer(
            positive_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        negative_enc = self.tokenizer(
            negative_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'anchor': {k: v.squeeze(0) for k, v in anchor_enc.items()},
            'positive': {k: v.squeeze(0) for k, v in positive_enc.items()},
            'negative': {k: v.squeeze(0) for k, v in negative_enc.items()}
        }