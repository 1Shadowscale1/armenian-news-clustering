"""
Пример использования пакета для кластеризации армянских новостей
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Добавляем путь к проекту
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ArmenianNews import (
    ArmenianNewsDataLoader,
    ArmenianTextPreprocessor,
    ArmenianEmbeddingModel,
    ArmenianNERModel
)
from ArmenianNews.clustering.similarity import SimilarityCalculator
from ArmenianNews.clustering.clustering import NewsClustering
from ArmenianNews.clustering.analysis import ClusterAnalyzer
from ArmenianNews.utils.visualization import NewsVisualization


def pipeline():
    # Конфигурация
    NUMBER_OF_ARTICLES_FROM_ONE_SOURCE = 100
    file_paths = [
        '/kaggle/input/armenian-political-news/armeniatoday.csv',
        '/kaggle/input/armenian-political-news/armenpress.csv',
        '/kaggle/input/armenian-political-news/hetq.csv',
        '/kaggle/input/armenian-political-news/sputnik.csv',
        '/kaggle/input/armenian-political-news/tert.csv'
    ]

    # 1. Загрузка данных
    print("Loading data...")
    data_loader = ArmenianNewsDataLoader(sample_size=NUMBER_OF_ARTICLES_FROM_ONE_SOURCE)
    df = data_loader.load_data_optimized(file_paths)

    # 2. Препроцессинг
    print("Preprocessing data...")
    preprocessor = ArmenianTextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, date_column='date_time')

    # Получаем подготовленные тексты
    input_texts = df['full_text'].tolist()
    n_articles = len(input_texts)
    print(f"Processing {n_articles} articles...")

    # 3. Получение эмбеддингов
    print("Computing embeddings...")
    embedding_model = ArmenianEmbeddingModel()
    embeddings = embedding_model.get_embeddings_batch(input_texts)
    dates = df['date_time'].tolist()

    print("Embeddings computed successfully!")

    # 4. Извлечение именованных сущностей
    print("Extracting named entities...")
    ner_model = ArmenianNERModel()
    all_entities = ner_model.get_named_entities_batch(input_texts)

    print("Named entities extracted successfully")

    # 5. Вычисление матрицы схожести
    print("Computing similarity matrix...")
    similarity_calculator = SimilarityCalculator()
    similarity_matrix, semantic_matrix, entity_matrix, time_matrix, length_matrix = similarity_calculator.compute_similarity_matrix(
        embeddings=embeddings,
        all_entities=all_entities,
        dates=dates,
        texts=input_texts
    )

    print("Similarity matrix computed successfully")

    # 6. Кластеризация
    print("\n=== Improved Clustering based on Combined Similarity ===\n")
    clustering = NewsClustering(min_cluster_size=3)
    cluster_labels = clustering.optimized_clustering(similarity_matrix)

    print(f"Initial clustering completed. Found {len(np.unique(cluster_labels))} clusters")
    print(f"Cluster distribution: {np.bincount(cluster_labels + 1)}")

    # 7. Анализ кластеров
    print("\nAnalyzing clusters...")
    analyzer = ClusterAnalyzer()
    clusters = analyzer.analyze_combined_clusters(
        df=df,
        cluster_labels=cluster_labels,
        input_texts=input_texts,
        dates=dates,
        ner_model=ner_model
    )

    # 8. Пост-обработка кластеров
    print("Post-processing clusters...")
    clusters = clustering.post_process_clusters(clusters, similarity_matrix)

    # 9. Визуализация
    print("\nCreating visualization...")
    visualizer = NewsVisualization()

    # Обновляем метки кластеров для визуализации
    updated_cluster_labels = np.full(len(df), -1)
    for cluster_id, cluster_info in clusters.items():
        if cluster_id != -1:
            for item in cluster_info['items']:
                updated_cluster_labels[item['original_index']] = cluster_id

    # Создаем визуализацию
    fig = visualizer.create_cluster_visualization(
        embeddings=embeddings.numpy(),
        cluster_labels=updated_cluster_labels,
        df=df
    )
    plt.show()

    # 12. Статистика кластеризации
    noise_count = np.sum(updated_cluster_labels == -1)
    valid_clusters = len([c for c in clusters.keys() if c != -1 and len(clusters[c]['items']) >= 2])

    print(f"\n📊 IMPROVED CLUSTERING SUMMARY:")
    print(f"   Total articles: {len(df)}")
    print(f"   Valid clusters: {valid_clusters}")
    print(f"   Noise articles: {noise_count}")
    print(f"   Articles in clusters: {len(df) - noise_count}")


    # 13. Создание отображения кластеров на новости
    print("\n📊 Creating cluster-news mapping...")
    cluster_news_mapping = analyzer.create_cluster_news_mapping(
        clusters=clusters,
        output_file='/kaggle/working/cluster_news_mapping.json'
    )