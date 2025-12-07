"""
Пример использования пакета для кластеризации армянских новостей
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from armenian_news_clustering import (
    ArmenianNewsDataLoader,
    ArmenianTextPreprocessor,
    ArmenianEmbeddingModel,
    ArmenianNERModel,
    SimilarityCalculator,
    NewsClustering,
    ClusterAnalyzer,
    NewsVisualization
)


def main():
    # Инициализация компонентов
    print("Initializing components...")

    data_loader = ArmenianNewsDataLoader(sample_size=100)
    preprocessor = ArmenianTextPreprocessor()
    embedding_model = ArmenianEmbeddingModel()
    ner_model = ArmenianNERModel()
    similarity_calculator = SimilarityCalculator()
    clustering = NewsClustering(min_cluster_size=3)
    analyzer = ClusterAnalyzer()
    visualizer = NewsVisualization()

    # Загрузка данных
    print("Loading data...")
    file_paths = [
        'path/to/armeniatoday.csv',
        'path/to/armenpress.csv',
        'path/to/hetq.csv',
        'path/to/sputnik.csv',
        'path/to/tert.csv'
    ]

    df = data_loader.load_data_optimized(file_paths)

    # Препроцессинг
    print("Preprocessing data...")
    df = preprocessor.preprocess_dataframe(df)

    # Получение эмбеддингов
    print("Computing embeddings...")
    embeddings = embedding_model.get_embeddings_batch(df['full_text'].tolist())

    # Извлечение сущностей
    print("Extracting named entities...")
    all_entities = ner_model.get_named_entities_batch(df['full_text'].tolist())

    # Вычисление схожести
    print("Calculating similarity matrix...")
    similarity_matrix, _, _, _, _ = similarity_calculator.compute_similarity_matrix(
        embeddings=embeddings,
        all_entities=all_entities,
        dates=df['date_time'].tolist(),
        texts=df['full_text'].tolist()
    )

    # Кластеризация
    print("Clustering articles...")
    cluster_labels = clustering.optimized_clustering(similarity_matrix)

    # Анализ кластеров
    print("Analyzing clusters...")
    clusters = analyzer.analyze_combined_clusters(
        df=df,
        cluster_labels=cluster_labels,
        input_texts=df['full_text'].tolist(),
        dates=df['date_time'].tolist(),
        ner_model=ner_model
    )

    # Пост-обработка кластеров
    clusters = clustering.post_process_clusters(clusters, similarity_matrix)

    # Сводка
    summary = analyzer.get_cluster_summary(clusters)
    print(summary)

    # Визуализация
    print("Creating visualizations...")
    fig = visualizer.create_cluster_visualization(
        embeddings=embeddings.numpy(),
        cluster_labels=cluster_labels,
        df=df
    )

    # Сохранение результатов
    mapping = analyzer.create_cluster_news_mapping(
        clusters=clusters,
        output_file='cluster_mapping.json'
    )

    print("Done!")


if __name__ == "__main__":
    main()