"""
Armenian News Clustering Package
"""

from .data_loader import ArmenianNewsDataLoader
from .preprocessing import ArmenianTextPreprocessor
from .triplet_generation import create_all_triplets
from .models.embedding_model import ArmenianEmbeddingModel
from .models.ner_model import ArmenianNERModel
from .clustering.similarity import SimilarityCalculator
from .clustering.clustering import NewsClustering
from .clustering.analysis import ClusterAnalyzer
from .utils.visualization import NewsVisualization

__version__ = "0.1.1"
__author__ = "Ilya Danilov"