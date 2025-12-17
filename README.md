# Armenian News Clustering

Пакет для кластеризации новостных статей на армянском языке с использованием современных методов NLP и машинного обучения

## Особенности

- Загрузка и препроцессинг армянских новостных данных
- Получение эмбеддингов с использованием предобученных моделей
- Извлечение именованных сущностей на армянском языке
- Генерация триплетов для дообучения моделей
- Кластеризация статей на основе семантической схожести
- Визуализация результатов кластеризации

## Установка

```bash
# Установка из исходного кода
git clone https://github.com/1Shadowscale1/armenian-news-clustering.git
cd armenian-news-clustering
pip install -e .

# Или установка зависимостей
pip install -r requirements.txt

#Импорт
from ArmenianNews import (
    ArmenianNewsDataLoader,
    TextPreprocessor,
    EmbeddingModel,
    NERModel
)
from ArmenianNews.clustering.similarity import SimilarityCalculator
from ArmenianNews.clustering.clustering import NewsClustering
from ArmenianNews.clustering.analysis import ClusterAnalyzer
from ArmenianNews.utils.visualization import NewsVisualization
from ArmenianNews.utils.link_parser import LinkParser
from ArmenianNews.basic import Pipeline