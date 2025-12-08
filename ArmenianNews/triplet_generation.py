import numpy as np
import random
from typing import List, Dict, Optional
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TripletGenerator:
    """Генерация триплетов для обучения моделей"""

    def __init__(self, n_triplets: int = 1000, similarity_threshold: float = 0.6):
        self.n_triplets = n_triplets
        self.similarity_threshold = similarity_threshold

    def create_semantic_triplets_tfidf(self, texts: List[str]) -> List[Dict]:
        """
        Создание триплетов на основе семантической схожести с использованием TF-IDF
        """
        triplets = []

        # Создаем TF-IDF векторизатор
        vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except:
            return triplets

        # Вычисляем косинусную схожесть
        similarity_matrix = cosine_similarity(tfidf_matrix)

        for i in range(min(self.n_triplets, len(texts) // 3)):
            anchor_idx = np.random.randint(0, len(texts))

            # Positive: наиболее схожие статьи (кроме самой себя)
            similarities = list(enumerate(similarity_matrix[anchor_idx]))
            similarities.sort(key=lambda x: x[1], reverse=True)

            positive_candidates = [
                idx for idx, sim in similarities[1:6]  # Топ-5 наиболее схожих
                if sim >= self.similarity_threshold and idx != anchor_idx
            ]

            if not positive_candidates:
                continue

            positive_idx = random.choice(positive_candidates)

            # Negative: наименее схожие статьи
            negative_candidates = [
                idx for idx, sim in similarities[-10:]  # 10 наименее схожих
                if idx != anchor_idx and idx != positive_idx and sim < 0.1
            ]

            if not negative_candidates:
                # Если нет явно негативных, берем случайные с низкой схожестью
                negative_candidates = [
                    idx for idx in range(len(texts))
                    if idx != anchor_idx and idx != positive_idx
                       and similarity_matrix[anchor_idx][idx] < 0.1
                ]

            if not negative_candidates:
                continue

            negative_idx = random.choice(negative_candidates)

            triplets.append({
                'anchor': anchor_idx,
                'positive': positive_idx,
                'negative': negative_idx,
                'similarity_anchor_positive': similarity_matrix[anchor_idx][positive_idx],
                'similarity_anchor_negative': similarity_matrix[anchor_idx][negative_idx],
                'type': 'semantic_tfidf'
            })

        return triplets

    def create_length_based_triplets(self, texts: List[str], length_ratio: float = 0.7) -> List[Dict]:
        """
        Создание триплетов на основе длины текста
        """
        triplets = []

        # Вычисляем длины текстов
        text_lengths = [len(text.split()) for text in texts]

        for _ in range(min(self.n_triplets, len(texts) // 3)):
            anchor_idx = np.random.randint(0, len(texts))
            anchor_length = text_lengths[anchor_idx]

            # Positive: статьи схожей длины
            positive_candidates = [
                idx for idx, length in enumerate(text_lengths)
                if idx != anchor_idx and
                   min(length, anchor_length) / max(length, anchor_length) >= length_ratio
            ]

            if not positive_candidates:
                continue

            positive_idx = random.choice(positive_candidates)

            # Negative: статьи значительно отличающейся длины
            negative_candidates = [
                idx for idx, length in enumerate(text_lengths)
                if idx != anchor_idx and idx != positive_idx and
                   min(length, anchor_length) / max(length, anchor_length) < 0.3
            ]

            if not negative_candidates:
                continue

            negative_idx = random.choice(negative_candidates)

            triplets.append({
                'anchor': anchor_idx,
                'positive': positive_idx,
                'negative': negative_idx,
                'length_anchor': anchor_length,
                'length_positive': text_lengths[positive_idx],
                'length_negative': text_lengths[negative_idx],
                'type': 'length_based'
            })

        return triplets

    def create_hybrid_triplets(self, texts: List[str], strategies: List[str] = None) -> List[Dict]:
        """
        Создание триплетов с использованием нескольких стратегий
        """
        if strategies is None:
            strategies = ['semantic', 'length']

        all_triplets = []
        triplets_per_strategy = self.n_triplets // len(strategies)

        if 'semantic' in strategies:
            semantic_triplets = self.create_semantic_triplets_tfidf(texts, triplets_per_strategy)
            all_triplets.extend(semantic_triplets)
            print(f"Created {len(semantic_triplets)} semantic triplets")

        # Перемешиваем триплеты
        random.shuffle(all_triplets)
        return all_triplets[:self.n_triplets]

    def analyze_triplets_quality(self: List[str], texts, dates=None):
        """
        Анализ качества созданных триплетов

        Args:
            self: список триплетов
            texts: список текстов
            dates: список дат (опционально)
        """
        print(f"Total triplets: {len(self)}")

        # Анализ по типам
        type_counts = {}
        for triplet in self:
            t_type = triplet.get('type', 'unknown')
            type_counts[t_type] = type_counts.get(t_type, 0) + 1

        print("Triplets by type:")
        for t_type, count in type_counts.items():
            print(f"  {t_type}: {count}")

        # Анализ длин текстов
        if 'length_anchor' in self[0]:
            avg_lengths = {
                'anchor': np.mean([t['length_anchor'] for t in self]),
                'positive': np.mean([t['length_positive'] for t in self]),
                'negative': np.mean([t['length_negative'] for t in self])
            }
            print(f"Average lengths: {avg_lengths}")

        # Анализ схожести для семантических триплетов
        if 'similarity_anchor_positive' in self[0]:
            avg_sim_pos = np.mean([t['similarity_anchor_positive'] for t in self])
            avg_sim_neg = np.mean([t['similarity_anchor_negative'] for t in self])
            print(f"Average similarity - anchor-positive: {avg_sim_pos:.3f}")
            print(f"Average similarity - anchor-negative: {avg_sim_neg:.3f}")


def create_all_triplets(texts: List[str], n_triplets: int = 800) -> List[Dict]:
    """
    Создание триплетов всеми методами
    """
    generator = TripletGenerator(n_triplets=n_triplets)
    return generator.create_hybrid_triplets(texts, strategies=['semantic', 'length'])