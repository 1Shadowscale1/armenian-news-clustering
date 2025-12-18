import numpy as np
import hdbscan
from typing import Dict, List

class NewsClustering:
    def __init__(self, min_cluster_size: int = 2, min_samples: int = 2):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    @staticmethod
    def evaluate_clustering_quality(labels: np.ndarray, similarity_matrix: np.ndarray) -> float:
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            return 0

        intra_cluster_similarity = 0
        cluster_count = 0

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < 2:
                continue

            # Внутрикластерная схожесть
            intra_sim = 0
            pair_count = 0
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    intra_sim += similarity_matrix[cluster_indices[i], cluster_indices[j]]
                    pair_count += 1

            if pair_count > 0:
                intra_cluster_similarity += intra_sim / pair_count
                cluster_count += 1

        if cluster_count > 0:
            intra_cluster_similarity /= cluster_count

        return intra_cluster_similarity

    def optimized_clustering(self, similarity_matrix: np.ndarray) -> np.ndarray:
        distance_matrix = 1 - similarity_matrix
        distance_matrix = distance_matrix.astype(np.float64)

        best_labels = None
        best_score = -1
        best_params = {}

        for min_cluster_size in [3, 4, 5]:
            for min_samples in [2, 3]:
                for cluster_selection_epsilon in [0.1, 0.2, 0.3]:
                    try:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric='precomputed',
                            cluster_selection_method='eom',
                            cluster_selection_epsilon=cluster_selection_epsilon
                        )

                        labels = clusterer.fit_predict(distance_matrix)

                        score = self.evaluate_clustering_quality(labels, similarity_matrix)

                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_params = {
                                'min_cluster_size': min_cluster_size,
                                'min_samples': min_samples,
                                'epsilon': cluster_selection_epsilon
                            }
                    except Exception as e:
                        print(f"Error with params {min_cluster_size}, {min_samples}, {cluster_selection_epsilon}: {e}")
                        continue

        print(f"Best clustering params: {best_params}, score: {best_score:.3f}")
        return best_labels

    def split_heterogeneous_cluster(self, cluster_info: Dict, similarity_matrix: np.ndarray,
                                    similarity_threshold: float = 0.5) -> List[Dict]:
        if len(cluster_info['items']) <= 2:
            return [cluster_info]

        indices = [item['original_index'] for item in cluster_info['items']]

        # Построение графа связности
        connections = {}
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if similarity_matrix[indices[i], indices[j]] >= similarity_threshold:
                    connections.setdefault(i, set()).add(j)
                    connections.setdefault(j, set()).add(i)

        # Поиск связных компонентов
        visited = set()
        subclusters = []

        for i in range(len(indices)):
            if i not in visited:
                component = []
                stack = [i]
                visited.add(i)

                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in connections.get(node, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)

                if component:
                    subcluster_items = [cluster_info['items'][idx] for idx in component]
                    subclusters.append({
                        'items': subcluster_items,
                        'common_entities': set(),
                        'date_range': None,
                        'entities_by_type': {'PER': set(), 'LOC': set(), 'ORG': set(), 'MISC': set()}
                    })

        return subclusters

    def post_process_clusters(self, clusters: Dict, similarity_matrix: np.ndarray,
                              min_intra_similarity: float = 0.6) -> Dict:
        """Пост-обработка для удаления неоднородных кластеров"""

        processed_clusters = {}
        new_cluster_id = 0

        for old_cluster_id, cluster_info in clusters.items():
            if old_cluster_id == -1:
                processed_clusters[-1] = cluster_info
                continue

            # Проверка средней внутрикластерной схожести
            cluster_indices = [item['original_index'] for item in cluster_info['items']]

            if len(cluster_indices) < 2:
                processed_clusters.setdefault(-1, {'items': []})['items'].extend(cluster_info['items'])
                continue

            intra_similarity = 0
            pair_count = 0
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    intra_similarity += similarity_matrix[cluster_indices[i], cluster_indices[j]]
                    pair_count += 1

            avg_intra_similarity = intra_similarity / pair_count if pair_count > 0 else 0

            if avg_intra_similarity >= min_intra_similarity:
                processed_clusters[new_cluster_id] = cluster_info
                new_cluster_id += 1
            else:
                print(f"Splitting cluster {old_cluster_id} (avg similarity: {avg_intra_similarity:.3f})")
                split_clusters = self.split_heterogeneous_cluster(cluster_info, similarity_matrix)
                for split_cluster in split_clusters:
                    if len(split_cluster['items']) >= 2:
                        processed_clusters[new_cluster_id] = split_cluster
                        new_cluster_id += 1
                    else:
                        processed_clusters.setdefault(-1, {'items': []})['items'].extend(split_cluster['items'])

        return processed_clusters