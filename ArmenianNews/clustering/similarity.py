import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Set
from datetime import datetime
import re


class SimilarityCalculator:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def entity_similarity(ents1: Set[str], ents2: Set[str]) -> float:
        """Jaccard-сходство между множествами сущностей"""

        # Объединяем все непустые множества из каждого списка
        union_ents1 = set().union(*[ent for ent in ents1 if ent])
        union_ents2 = set().union(*[ent for ent in ents2 if ent])

        if not union_ents1 or not union_ents2:
            return 0

        intersection = len(union_ents1 & union_ents2)
        union_total = len(union_ents1 | union_ents2)

        return intersection / union_total if union_total > 0 else 0

    @staticmethod
    def temporal_distance(date1, date2, max_days: float = 0.4) -> float:
        """Строгая временная метрика"""
        if hasattr(date1, 'tzinfo') and date1.tzinfo is not None:
            date1 = date1.replace(tzinfo=None)
        if hasattr(date2, 'tzinfo') and date2.tzinfo is not None:
            date2 = date2.replace(tzinfo=None)

        delta = abs((date1 - date2).days)
        if delta > max_days:
            return 1.0
        return (delta / max_days) ** 1.5

    @staticmethod
    def length_difference_penalty(text1: str, text2: str, max_length_diff: int = 200) -> float:
        """Штраф за значительную разницу в длине текстов новостей"""
        len1 = len(text1)
        len2 = len(text2)

        length_diff = abs(len1 - len2)

        if length_diff == 0:
            return 0.0
        elif length_diff >= max_length_diff:
            return 1.0
        else:
            normalized_diff = length_diff / max_length_diff
            return normalized_diff ** 2

    def compute_similarity_matrix(self, embeddings: torch.Tensor, all_entities: List[Set[str]],
                                  dates: List, texts: List[str],
                                  alpha: float = 0.3, beta: float = 0.3,
                                  gamma: float = 0.2, delta: float = 0.2) -> Tuple[np.ndarray, ...]:
        """Вычисление матрицы схожести"""
        n_articles = len(embeddings)

        embeddings_gpu = embeddings.to(self.device)

        # Вычисление косинусной схожести
        embeddings_normalized = F.normalize(embeddings_gpu, p=2, dim=1)
        cosine_sim_matrix = torch.mm(embeddings_normalized, embeddings_normalized.T)

        # Создание матрицы схожести сущностей
        entity_sim_matrix = torch.zeros((n_articles, n_articles), device=self.device)

        for i in range(n_articles):
            for j in range(i + 1, n_articles):
                ent_sim = self.entity_similarity(all_entities[i], all_entities[j])
                entity_sim_matrix[i, j] = ent_sim
                entity_sim_matrix[j, i] = ent_sim

        # Создание временной матрицы
        time_matrix = torch.zeros((n_articles, n_articles), device=self.device)

        for i in range(n_articles):
            for j in range(i + 1, n_articles):
                time_penalty = self.temporal_distance(dates[i], dates[j])
                time_matrix[i, j] = time_penalty
                time_matrix[j, i] = time_penalty

        # Создание матрицы штрафов за разницу в длине
        length_matrix = torch.zeros((n_articles, n_articles), device=self.device)

        for i in range(n_articles):
            for j in range(i + 1, n_articles):
                length_penalty = self.length_difference_penalty(texts[i], texts[j])
                length_matrix[i, j] = length_penalty
                length_matrix[j, i] = length_penalty

        # Комбинирование всех компонентов
        semantic_component = alpha * cosine_sim_matrix
        entity_component = beta * (entity_sim_matrix ** 2)
        time_component = gamma * (time_matrix ** 2)
        length_component = delta * length_matrix

        similarity_matrix = semantic_component + entity_component - time_component - length_component
        similarity_matrix = torch.clamp(similarity_matrix, min=0.0)

        # Применение пороговых значений
        min_semantic_threshold = 0.3
        min_entity_threshold = 0.1

        mask_low_semantic = cosine_sim_matrix < min_semantic_threshold
        mask_low_entity = entity_sim_matrix < min_entity_threshold
        mask_combined = mask_low_semantic | mask_low_entity

        similarity_matrix[mask_combined] = 0.0

        # Округление и перенос на CPU
        similarity_matrix = torch.round(similarity_matrix * 1000) / 1000

        # Перенос результатов на CPU
        similarity_matrix_cpu = similarity_matrix.cpu().numpy()
        semantic_matrix_cpu = cosine_sim_matrix.cpu().numpy()
        entity_matrix_cpu = entity_sim_matrix.cpu().numpy()
        time_matrix_cpu = time_matrix.cpu().numpy()
        length_matrix_cpu = length_matrix.cpu().numpy()

        return similarity_matrix_cpu, semantic_matrix_cpu, entity_matrix_cpu, time_matrix_cpu, length_matrix_cpu