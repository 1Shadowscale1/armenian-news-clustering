import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import json


class ClusterAnalyzer:
    """Анализ кластеров новостных статей"""

    def __init__(self):
        pass

    def analyze_combined_clusters(self, df: pd.DataFrame, cluster_labels: np.ndarray,
                                  input_texts: List[str], dates: List,
                                  ner_model: Any) -> Dict:
        """Анализ кластеров с комбинированной метрикой"""
        clusters = {}

        # Предварительное вычисление сущностей
        print("🔍 Precomputing named entities for all texts...")
        all_entities_batch = ner_model.get_named_entities_batch(input_texts)

        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = {
                    'items': [],
                    'common_entities': set(),
                    'date_range': None,
                    'entities_by_type': {'PER': set(), 'LOC': set(), 'ORG': set(), 'MISC': set()}
                }

            entities_set = all_entities_batch[i] if i < len(all_entities_batch) else set()
            clusters[label]['common_entities'].update(entities_set)

            # Группировка сущностей по типам
            for entity in entities_set:
                if ':' in entity:
                    entity_type, entity_text = entity.split(':', 1)
                    if entity_type in clusters[label]['entities_by_type']:
                        clusters[label]['entities_by_type'][entity_type].add(entity)
                else:
                    clusters[label]['entities_by_type']['MISC'].add(f"MISC:{entity}")

            clusters[label]['items'].append({
                'text': input_texts[i],
                'title': df.iloc[i]['title'],
                'date': dates[i],
                'original_index': i,
                'entities': entities_set
            })

        # Вычисление временного диапазона для каждого кластера
        for label, cluster_data in clusters.items():
            if cluster_data['items']:
                cluster_dates = [item['date'] for item in cluster_data['items']]
                cluster_data['date_range'] = {
                    'min': min(cluster_dates),
                    'max': max(cluster_dates),
                    'span_days': (max(cluster_dates) - min(cluster_dates)).days
                }

        return clusters

    def analyze_cluster_quality(self, clusters: Dict, similarity_matrix: np.ndarray) -> Dict:
        """Анализ качества кластеров"""
        quality_metrics = {
            'total_clusters': len([k for k in clusters.keys() if k != -1]),
            'total_articles': sum(len(c['items']) for c in clusters.values()),
            'noise_articles': len(clusters.get(-1, {}).get('items', [])),
            'cluster_sizes': {},
            'avg_intra_similarity': {}
        }

        for cluster_id, cluster_info in clusters.items():
            if cluster_id == -1:
                continue

            cluster_indices = [item['original_index'] for item in cluster_info['items']]
            quality_metrics['cluster_sizes'][cluster_id] = len(cluster_indices)

            # Вычисление внутрикластерной схожести
            intra_similarity = 0
            pair_count = 0
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    intra_similarity += similarity_matrix[cluster_indices[i], cluster_indices[j]]
                    pair_count += 1

            avg_similarity = intra_similarity / pair_count if pair_count > 0 else 0
            quality_metrics['avg_intra_similarity'][cluster_id] = avg_similarity

        return quality_metrics

    def create_cluster_news_mapping(self, clusters: Dict, output_file: str = None) -> Dict:
        """Создание отображения кластеров на новостные статьи"""
        cluster_news_mapping = {}

        for cluster_id, cluster_info in clusters.items():
            # Пропускаем шумовые кластеры (-1)
            if int(cluster_id) == -1:
                continue

            news_ids = [item['original_index'] for item in cluster_info['items']]
            cluster_news_mapping[int(cluster_id)] = news_ids

        # Добавление шумовых статей как отдельного кластера
        if -1 in clusters:
            noise_articles = [item['original_index'] for item in clusters[-1]['items']]
            if noise_articles:
                cluster_news_mapping[-1] = noise_articles

        # Сохранение в файл, если указан путь
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cluster_news_mapping, f, indent=2, ensure_ascii=False)
            print(f"Cluster-news mapping saved to: {output_file}")

        return cluster_news_mapping

    def get_cluster_summary(self, clusters: Dict) -> str:
        """Получение текстовой сводки по кластерам"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("CLUSTER ANALYSIS SUMMARY:")
        summary_lines.append("=" * 80)

        valid_clusters = [k for k in clusters.keys() if k != -1]
        noise_count = len(clusters.get(-1, {}).get('items', []))
        total_in_clusters = sum(len(clusters[k]['items']) for k in valid_clusters)

        summary_lines.append(f"\n📊 Total clusters: {len(valid_clusters)}")
        summary_lines.append(f"📝 Articles in clusters: {total_in_clusters}")
        summary_lines.append(f"🎭 Noise articles: {noise_count}")
        summary_lines.append(f"📈 Total articles: {total_in_clusters + noise_count}")

        summary_lines.append("\n🔍 Cluster Details:")
        for cluster_id in sorted(valid_clusters):
            cluster_info = clusters[cluster_id]
            summary_lines.append(f"\n  Cluster {cluster_id}:")
            summary_lines.append(f"    📝 Size: {len(cluster_info['items'])} articles")

            # Ключевые сущности
            key_entities = []
            for entity_type in ['PER', 'LOC', 'ORG']:
                entities = list(cluster_info['entities_by_type'][entity_type])
                if entities:
                    key_entities.extend(entities[:2])

            if key_entities:
                summary_lines.append(f"    🔑 Key entities: {', '.join(key_entities[:5])}")

            # Пример заголовка
            if cluster_info['items']:
                first_item = cluster_info['items'][0]
                summary_lines.append(f"    📰 Sample: '{first_item['title'][:70]}...'")

        return "\n".join(summary_lines)