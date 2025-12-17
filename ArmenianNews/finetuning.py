import gc
import os
import warnings
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ArmenianNews import (
    ArmenianNewsDataLoader,
    TextPreprocessor
)
from .models.dataset import OptimizedTripletDataset
from .models.embedding_model import EmbeddingModel, TripletLoss

warnings.filterwarnings('ignore')


class FineTuningManager:
    """Менеджер для fine-tuning модели эмбеддингов на армянских новостях"""

    def __init__(self, model_name: str = "Metric-AI/armenian-text-embeddings-1",
                 device: str = None):
        """
        Инициализация менеджера fine-tuning

        Args:
            model_name: Название предобученной модели
            device: Устройство для обучения ('cuda' или 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        # Инициализация модели и токенизатора
        print(f"Initializing fine-tuning on {self.device}...")
        self.model = EmbeddingModel(model_name, device)
        self.tokenizer = self.model.tokenizer

        # Настройки для экономии памяти
        torch.backends.cudnn.benchmark = True
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_training_data(self, file_paths: List[str], sample_size: int = 100) -> Dict:
        """
        Подготовка данных для обучения

        Args:
            file_paths: Список путей к CSV файлам
            sample_size: Количество статей из каждого источника
        """
        print("Loading data")

        # Загрузка данных
        data_loader = ArmenianNewsDataLoader.load_data(file_paths, sample_size)
        df = data_loader

        # Преобразование дат
        print("Converting dates")
        df['date_time'] = df['date_time'].apply(TextPreprocessor.convert_armenian_date)
        df = df.dropna(subset=['date_time']).reset_index(drop=True)

        # Подготовка текстов
        print("Preparing texts")
        input_texts = []
        for idx, row in df.iterrows():
            text = f"{row['title']}. {row['text']}"
            # Обрезаем очень длинные тексты
            if len(text) > 1000:
                text = text[:1000]
            input_texts.append(text)

        print(f"Processing {len(input_texts)} articles...")

        return {
            'dataframe': df,
            'texts': input_texts,
            'dates': df['date_time'].tolist()
        }

    def create_triplets(self, df, input_texts: List[str], dates: List,
                        n_triplets: int = 400) -> List[Dict]:
        """
        Создание триплетов для обучения

        Args:
            df: DataFrame с данными
            input_texts: Список текстов
            dates: Список дат
            n_triplets: Количество триплетов для создания
        """
        print("Creating triplets...")

        # Определяем максимальное количество триплетов
        max_triplets = min(n_triplets, len(df) // 3)
        if max_triplets < 10:
            warnings.warn(f"Not enough data for triplets: only {len(df)} articles")
            max_triplets = max(10, len(df) // 2)

        # Используем функцию из triplet_generation
        from .triplet_generation import TripletGenerator
        generator = TripletGenerator(n_triplets=max_triplets)

        # Создаем триплеты различными стратегиями
        semantic_triplets = generator.create_semantic_triplets_tfidf(input_texts)
        length_triplets = generator.create_length_based_triplets(input_texts)

        # Объединяем и перемешиваем триплеты
        all_triplets = semantic_triplets + length_triplets

        print(f"Created {len(all_triplets)} triplets")

        # Выводим статистику
        semantic_count = len([t for t in all_triplets if t.get('type') == 'semantic_tfidf'])
        length_count = len([t for t in all_triplets if t.get('type') == 'length_based'])

        print(f"   - Semantic triplets: {semantic_count}")
        print(f"   - Length-based triplets: {length_count}")

        return all_triplets

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    accumulation_steps: int = 4) -> float:
        """
        Обучение на одной эпохе

        Args:
            dataloader: DataLoader с данными
            optimizer: Оптимизатор
            accumulation_steps: Количество шагов для накопления градиентов

        Returns:
            Среднее значение loss на эпохе
        """
        self.model.model.train()
        total_loss = 0
        loss_fn = TripletLoss(margin=1.0)

        optimizer.zero_grad()

        with tqdm(dataloader, desc="Training", unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Перенос данных на устройство
                anchor_inputs = {k: v.to(self.device) for k, v in batch['anchor'].items()}
                positive_inputs = {k: v.to(self.device) for k, v in batch['positive'].items()}
                negative_inputs = {k: v.to(self.device) for k, v in batch['negative'].items()}

                # Получение эмбеддингов
                with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    anchor_outputs = self.model.model(**anchor_inputs)
                    positive_outputs = self.model.model(**positive_inputs)
                    negative_outputs = self.model.model(**negative_inputs)

                    # Pooling эмбеддингов
                    anchor_embeddings = self.model.average_pool(
                        anchor_outputs.last_hidden_state,
                        anchor_inputs['attention_mask']
                    )
                    positive_embeddings = self.model.average_pool(
                        positive_outputs.last_hidden_state,
                        positive_inputs['attention_mask']
                    )
                    negative_embeddings = self.model.average_pool(
                        negative_outputs.last_hidden_state,
                        negative_inputs['attention_mask']
                    )

                    # Нормализация
                    anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
                    positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
                    negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

                    # Вычисление loss
                    loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                    loss = loss / accumulation_steps

                # Backward pass
                loss.backward()

                # Обновление весов после accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                pbar.set_postfix({'loss': loss.item() * accumulation_steps})

        return total_loss / len(dataloader)

    def fine_tune(self, file_paths: List[str], output_dir: str,
                  num_epochs: int = 3, batch_size: int = 4,
                  learning_rate: float = 1e-5, n_triplets: int = 400) -> Dict:
        """
        Основная функция fine-tuning

        Args:
            file_paths: Список путей к данным
            output_dir: Директория для сохранения модели
            num_epochs: Количество эпох обучения
            batch_size: Размер батча
            learning_rate: Скорость обучения
            n_triplets: Количество триплетов
        """
        print("Starting fine-tuning process...")

        # Подготовка данных
        data = self.prepare_training_data(file_paths)
        df = data['dataframe']
        input_texts = data['texts']
        dates = data['dates']

        # Создание триплетов
        triplets = self.create_triplets(df, input_texts, dates, n_triplets)

        if len(triplets) < 10:
            raise ValueError(f"Not enough triplets created: {len(triplets)}. Need at least 10.")

        # Создание dataset и dataloader
        print("Creating dataset and dataloader...")
        dataset = OptimizedTripletDataset(
            texts=input_texts,
            triplets=triplets,
            tokenizer=self.tokenizer,
            max_length=128
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

        # Оптимизатор
        optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Обучение
        print(f"Training for {num_epochs} epochs...")
        training_history = {
            'epochs': [],
            'losses': [],
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'triplets_used': len(triplets)
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            avg_loss = self.train_epoch(dataloader, optimizer, accumulation_steps=4)
            training_history['epochs'].append(epoch + 1)
            training_history['losses'].append(avg_loss)

            print(f"Average loss: {avg_loss:.4f}")

            # Очистка памяти
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # Сохранение модели
        print(f"Saving model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        self.model.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print("Fine-tuning completed successfully")

        return training_history

    def evaluate_fine_tuning(self, test_file_paths: List[str],
                             original_model_name: str = None) -> Dict:
        """
        Оценка эффективности fine-tuning

        Args:
            test_file_paths: Пути к тестовым данным
            original_model_name: Название исходной модели для сравнения

        Returns:
            Словарь с метриками оценки
        """
        print("Evaluating fine-tuning results")

        # Загрузка тестовых данных
        test_data = self.prepare_training_data(test_file_paths, sample_size=50)
        test_texts = test_data['texts']

        # Получение эмбеддингов до и после fine-tuning
        print("Getting embeddings from fine-tuned model")
        ft_embeddings = self.model.get_embeddings_batch(test_texts, batch_size=8)

        # Если указана исходная модель, сравниваем с ней
        comparison_results = {}
        if original_model_name:
            print(f"Comparing with original model: {original_model_name}")
            original_model = EmbeddingModel(original_model_name, self.device)
            original_embeddings = original_model.get_embeddings_batch(test_texts, batch_size=8)

            # Вычисление косинусной схожести между эмбеддингами
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(ft_embeddings.numpy(), original_embeddings.numpy())
            avg_similarity = similarity_matrix.diagonal().mean()

            comparison_results = {
                'avg_similarity_to_original': avg_similarity,
                'similarity_matrix': similarity_matrix
            }

        # Оценка качества кластеризации на тестовых данных
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Кластеризация
        n_clusters = min(5, len(test_texts) // 3)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(ft_embeddings.numpy())
            silhouette = silhouette_score(ft_embeddings.numpy(), cluster_labels)

            comparison_results['clustering_quality'] = {
                'silhouette_score': silhouette,
                'n_clusters': n_clusters,
                'cluster_distribution': np.bincount(cluster_labels)
            }

        print("Evaluation completed!")
        return comparison_results


# Функция для быстрого запуска fine-tuning
def fine_tune_armenian_model(file_paths: List[str], output_dir: str = "./fine_tuned_model",
                             model_name: str = "Metric-AI/armenian-text-embeddings-1",
                             num_epochs: int = 3, batch_size: int = 4,
                             learning_rate: float = 1e-5, n_triplets: int = 400) -> Dict:
    """
    Быстрый запуск fine-tuning модели

    Args:
        file_paths: Список путей к CSV файлам с данными
        output_dir: Директория для сохранения модели
        model_name: Название предобученной модели
        num_epochs: Количество эпох обучения
        batch_size: Размер батча
        learning_rate: Скорость обучения
        n_triplets: Количество триплетов для создания
    """
    # Инициализация менеджера
    ft_manager = FineTuningManager(model_name=model_name)

    # Запуск fine-tuning
    results = ft_manager.fine_tune(
        file_paths=file_paths,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_triplets=n_triplets
    )

    return results


# Основная функция для запуска из командной строки
def finetune():
    """
    Основной процесс fine-tuning
    """
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune Armenian text embedding model')
    parser.add_argument('--data_paths', nargs='+', required=True,
                        help='Paths to CSV files with Armenian news')
    parser.add_argument('--output_dir', default='./fine_tuned_model',
                        help='Directory to save fine-tuned model')
    parser.add_argument('--model_name', default='Metric-AI/armenian-text-embeddings-1',
                        help='Name of the pretrained model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--n_triplets', type=int, default=400,
                        help='Number of triplets to create')

    args = parser.parse_args()

    # Запуск fine-tuning
    print("Starting Armenian News Model Fine-Tuning")
    print(f"Model: {args.model_name}")
    print(f"Data: {len(args.data_paths)} files")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    try:
        results = fine_tune_armenian_model(
            file_paths=args.data_paths,
            output_dir=args.output_dir,
            model_name=args.model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_triplets=args.n_triplets
        )

        print("\n" + "=" * 50)
        print("FINE-TUNING COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Model saved to: {args.output_dir}")
        print(f"Training history: {results}")

    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()