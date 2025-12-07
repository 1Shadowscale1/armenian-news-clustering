import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Dict


class NewsVisualization:
    """Визуализация результатов кластеризации новостей"""

    def __init__(self):
        pass

    def create_cluster_visualization(self, embeddings: np.ndarray, cluster_labels: np.ndarray,
                                     df: pd.DataFrame, figsize: tuple = (20, 8)) -> plt.Figure:
        """Создание визуализации кластеров"""

        # Используем исходные эмбеддинги для t-SNE
        tsne_original = TSNE(n_components=2, random_state=42,
                             perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne_original.fit_transform(embeddings)

        # PCA для уменьшения размерности
        pca = PCA(n_components=2)
        embeddings_2d_pca = pca.fit_transform(embeddings)

        # Создание DataFrame для визуализации
        viz_df = pd.DataFrame({
            'index': range(len(df)),
            'title': df['title'],
            'date': df['date_time'],
            'cluster': cluster_labels,
            'x_tsne': embeddings_2d[:, 0],
            'y_tsne': embeddings_2d[:, 1],
            'x_pca': embeddings_2d_pca[:, 0],
            'y_pca': embeddings_2d_pca[:, 1],
            'is_noise': cluster_labels == -1
        })

        # Создание двух визуализаций
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Визуализация 1: t-SNE на исходных эмбеддингах
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

        # t-SNE визуализация
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:
                cluster_mask = viz_df['cluster'] == cluster_id
                ax1.scatter(viz_df[cluster_mask]['x_tsne'], viz_df[cluster_mask]['y_tsne'],
                            c='gray', s=30, alpha=0.3, label='Noise')
            else:
                cluster_mask = viz_df['cluster'] == cluster_id
                if cluster_mask.sum() > 0:
                    ax1.scatter(viz_df[cluster_mask]['x_tsne'], viz_df[cluster_mask]['y_tsne'],
                                c=[colors[i]], s=80, label=f'Cluster {cluster_id} (n={cluster_mask.sum()})',
                                alpha=0.8, edgecolors='black')

        ax1.set_title('t-SNE Visualization of News Clusters', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Визуализация 2: PCA
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:
                cluster_mask = viz_df['cluster'] == cluster_id
                ax2.scatter(viz_df[cluster_mask]['x_pca'], viz_df[cluster_mask]['y_pca'],
                            c='gray', s=30, alpha=0.3, label='Noise')
            else:
                cluster_mask = viz_df['cluster'] == cluster_id
                if cluster_mask.sum() > 0:
                    ax2.scatter(viz_df[cluster_mask]['x_pca'], viz_df[cluster_mask]['y_pca'],
                                c=[colors[i]], s=80, label=f'Cluster {cluster_id} (n={cluster_mask.sum()})',
                                alpha=0.8, edgecolors='black')

        ax2.set_title('PCA Visualization of News Clusters', fontsize=14, fontweight='bold')
        ax2.set_xlabel('PCA Component 1')
        ax2.set_ylabel('PCA Component 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray,
                                cluster_labels: np.ndarray = None,
                                figsize: tuple = (10, 8)) -> plt.Figure:
        """Визуализация тепловой карты схожести"""
        fig, ax = plt.subplots(figsize=figsize)

        if cluster_labels is not None:
            # Сортируем матрицу по кластерам
            sort_idx = np.argsort(cluster_labels)
            sorted_matrix = similarity_matrix[sort_idx][:, sort_idx]
            sorted_labels = cluster_labels[sort_idx]

            im = ax.imshow(sorted_matrix, cmap='viridis', aspect='auto')

            # Добавляем разделители между кластерами
            unique_labels = np.unique(sorted_labels)
            for label in unique_labels:
                if label != -1:
                    idx = np.where(sorted_labels == label)[0]
                    if len(idx) > 0:
                        start = idx[0]
                        end = idx[-1]
                        ax.axhline(y=start - 0.5, color='white', linewidth=2)
                        ax.axvline(x=start - 0.5, color='white', linewidth=2)
                        ax.axhline(y=end + 0.5, color='white', linewidth=2)
                        ax.axvline(x=end + 0.5, color='white', linewidth=2)
        else:
            im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')

        ax.set_title('Similarity Matrix Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Article Index')
        ax.set_ylabel('Article Index')
        plt.colorbar(im, ax=ax, label='Similarity Score')

        return fig

    def plot_cluster_size_distribution(self, clusters: Dict, figsize: tuple = (10, 6)) -> plt.Figure:
        """Визуализация распределения размеров кластеров"""
        cluster_sizes = []
        cluster_ids = []

        for cluster_id, cluster_info in clusters.items():
            if cluster_id != -1:
                cluster_sizes.append(len(cluster_info['items']))
                cluster_ids.append(cluster_id)

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(range(len(cluster_sizes)), cluster_sizes)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Articles')
        ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels(cluster_ids)

        # Добавление значений на столбцы
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')

        return fig