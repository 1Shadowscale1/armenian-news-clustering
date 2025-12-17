"""
Armenian News Clustering Package
"""

from .clustering.analysis import ClusterAnalyzer
from .clustering.clustering import NewsClustering
from .clustering.similarity import SimilarityCalculator
from .models.dataset import NewsDataset
from .models.embedding_model import EmbeddingModel
from .models.ner_model import NERModel
from .utils.visualization import NewsVisualization
from .data_loader import ArmenianNewsDataLoader
from .preprocessing import TextPreprocessor
from .triplet_generation import create_all_triplets
from .basic import Pipeline

__version__ = "0.1.1"
__author__ = "Ilya Danilov"

# Импорт для удобного доступа ко всем классам и функциям
__all__ = [
    # Основные классы для загрузки данных
    "ArmenianNewsDataLoader",

    # Препроцессинг текста
    "TextPreprocessor",

    # Генерация триплетов
    "create_all_triplets",

    # Модели
    "EmbeddingModel",
    "NERModel",

    # Кластеризация
    "SimilarityCalculator",
    "NewsClustering",
    "ClusterAnalyzer",

    # Визуализация
    "NewsVisualization",

    # Пайплайн алгоритма
    "Pipeline",

    # Версия
    "__version__"
]