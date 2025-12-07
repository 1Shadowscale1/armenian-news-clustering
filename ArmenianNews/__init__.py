"""
Armenian News Clustering Package
"""

from .data_loader import ArmenianNewsDataLoader
from .preprocessing import ArmenianTextPreprocessor
from .triplet_generation import create_all_triplets
from .models.embedding_model import ArmenianEmbeddingModel
from .models.ner_model import ArmenianNERModel
from .clustering import SimilarityCalculator, NewsClustering, ClusterAnalyzer
from .utils.visualization import NewsVisualization

__version__ = "0.1.1"
__author__ = "Ilya Danilov"

# Импорт для удобного доступа ко всем классам и функциям
__all__ = [
    # Основные классы для загрузки данных
    "ArmenianNewsDataLoader",

    # Препроцессинг текста
    "ArmenianTextPreprocessor",

    # Генерация триплетов
    "create_all_triplets",

    # Модели
    "ArmenianEmbeddingModel",
    "ArmenianNERModel",

    # Кластеризация
    "SimilarityCalculator",
    "NewsClustering",
    "ClusterAnalyzer",

    # Визуализация
    "NewsVisualization",

    # Версия
    "__version__"
]