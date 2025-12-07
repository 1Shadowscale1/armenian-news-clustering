"""
Armenian News Clustering Package
"""

from .data_loader import load_data_optimized
from .preprocessing import convert_armenian_date, ArmenianTextPreprocessor
from .triplet_generation import create_all_triplets
from .models.embedding_model import ArmenianEmbeddingModel
from .models.ner_model import ArmenianNERModel
from ArmenianNews.clustering.clustering import compute_similarity_matrix
from ArmenianNews.clustering.clustering import NewsClustering

__version__ = "0.1.0"
__author__ = "Your Name"