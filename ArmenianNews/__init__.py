"""
Armenian News Clustering Package
"""

from .data_loader import ArmenianNewsDataLoader
from .preprocessing import ArmenianTextPreprocessor
from .triplet_generation import create_all_triplets
from .models.embedding_model import ArmenianEmbeddingModel
from .models.ner_model import ArmenianNERModel
from ArmenianNews.clustering.clustering import NewsClustering

__version__ = "0.1.1"
__author__ = "Ilya Danilov"