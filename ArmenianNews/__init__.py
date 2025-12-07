"""
Armenian News Clustering Package
"""

from .clustering.analysis import ClusterAnalyzer
from .clustering.clustering import NewsClustering
from .clustering.similarity import SimilarityCalculator
from .models.dataset import ArmenianNewsDataset
from .models.embedding_model import ArmenianEmbeddingModel
from .models.ner_model import ArmenianNERModel
from .utils.visualization import NewsVisualization
from .data_loader import ArmenianNewsDataLoader
from .preprocessing import ArmenianTextPreprocessor
from .triplet_generation import create_all_triplets

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