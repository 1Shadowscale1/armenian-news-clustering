import torch
import re
from transformers import AutoModelForTokenClassification, pipeline
from typing import List, Set, Dict
import warnings


class ArmenianNERModel:
    """Модель для распознавания именованных сущностей на армянском языке"""

    def __init__(self, model_name: str = "daviddallakyan2005/armenian-ner", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        # Загрузка модели NER
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if self.device == 'cuda' else -1
        )

    def batch_ner_processing(self, texts: List[str], batch_size: int = 16,
                             confidence_threshold: float = 0.7) -> List[Set[str]]:
        """Пакетная обработка текстов для извлечения сущностей"""
        all_entities = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                batch_preds = self.ner_pipeline(batch_texts)

                for preds in batch_preds:
                    normalized_entities = set()

                    if isinstance(preds, dict):
                        preds = [preds]
                    elif not isinstance(preds, list):
                        warnings.warn(f"Unexpected prediction format: {type(preds)}")
                        all_entities.append(set())
                        continue

                    for ent in preds:
                        # Пропускаем сущности с низкой уверенностью
                        if 'score' in ent and ent['score'] < confidence_threshold:
                            continue

                        entity_text = ent.get("word", "")
                        if not entity_text:
                            continue

                        if not self._is_valid_armenian_entity(entity_text):
                            continue

                        normalized_entity = self._normalize_armenian_entity(entity_text)

                        if not normalized_entity:
                            continue

                        # Обработка многословных сущностей
                        words = normalized_entity.split()
                        stemmed_words = []

                        for word in words:
                            stemmed_word = self._enhanced_armenian_stemmer(word)
                            if stemmed_word:
                                stemmed_words.append(stemmed_word)

                        if not stemmed_words:
                            continue

                        stemmed_entity = ' '.join(stemmed_words)
                        entity_type = ent.get('entity_group', 'MISC')

                        # Финальная валидация
                        if len(stemmed_entity) >= 2 and self._is_valid_armenian_entity(stemmed_entity):
                            final_entity = f"{entity_type}:{stemmed_entity}"
                            normalized_entities.add(final_entity)

                    all_entities.append(normalized_entities)

            except Exception as e:
                warnings.warn(f"Error in batch NER processing: {e}")
                all_entities.extend([set()] * len(batch_texts))

        return all_entities

    def _normalize_armenian_entity(self, entity_text: str) -> str:
        """Нормализация армянской сущности"""
        if not entity_text or not isinstance(entity_text, str):
            return ""

        # Удаление пунктуации и специальных символов
        entity_text = re.sub(r'[^\w\s\u0531-\u058F\u0561-\u0587]', '', entity_text)

        # Удаление лишних пробелов и нормализация
        entity_text = ' '.join(entity_text.split())
        entity_text = entity_text.strip()

        if not entity_text:
            return ""

        # Приведение к нижнему регистру
        entity_text = entity_text.lower()

        # Правила нормализации для армянского языка
        normalization_rules = {
            'եւ': 'և',
            'ու': 'ու',
            'ոՒ': 'ու',
            'ոու': 'ու',
        }

        for old, new in normalization_rules.items():
            entity_text = entity_text.replace(old, new)

        return entity_text

    def _is_valid_armenian_entity(self, entity_text: str) -> bool:
        """Проверка, является ли текст валидной армянской сущностью"""
        if not entity_text or not isinstance(entity_text, str):
            return False

        if len(entity_text) < 2:
            return False

        # Проверка наличия хотя бы одной армянской буквы
        if not re.search(r'[\u0531-\u058F\u0561-\u0587]', entity_text):
            return False

        # Фильтрация шаблонов шума
        noise_patterns = [
            r'^[ա-ֆ]{1,2}$',  # Одиночные или двойные буквы
            r'^\d+$',  # Только цифры
            r'^[ա-ֆ]\d+$',  # Буква, за которой следуют цифры
            r'^\d+[ա-ֆ]$',  # Цифры, за которыми следует буква
        ]

        for pattern in noise_patterns:
            if re.match(pattern, entity_text):
                return False

        # Фильтрация стоп-слов и шума
        armenian_stop_words = {
            'ը', 'ն', 'ում', 'իս', 'ով', 'ից', 'ան', 'ուն', 'յուն',
            'ություն', 'աց', 'եց', 'իկ', 'ակ', 'ային', 'ու', 'ի', 'է'
        }

        if entity_text in armenian_stop_words:
            return False

        return True

    def _enhanced_armenian_stemmer(self, word: str) -> str:
        """Улучшенный стеммер для армянского языка"""
        if not self._is_valid_armenian_entity(word):
            return ""

        suffixes = [
            'ը', 'ն', 'ում', 'իս', 'ով', 'ից', 'ան', 'ուն', 'յուն',
            'ություն', 'աց', 'եց', 'իկ', 'ակ', 'ային'
        ]

        suffixes.sort(key=len, reverse=True)

        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                word = word[:-len(suffix)]
                break

        # Финальная валидация после стемминга
        if not self._is_valid_armenian_entity(word):
            return ""

        return word

    def get_named_entities_batch(self, texts: List[str], min_entity_length: int = 3) -> List[Set[str]]:
        """Пакетное извлечение именованных сущностей"""
        entities_list = self.batch_ner_processing(texts)
        return self._filter_ner_results(entities_list, min_entity_length)

    def _filter_ner_results(self, entities_list: List[Set[str]], min_entity_length: int = 2) -> List[Set[str]]:
        """Пост-обработка результатов NER для удаления шума"""
        filtered_entities = []

        for entities in entities_list:
            filtered_set = set()
            for entity in entities:
                entity_parts = entity.split(':', 1)
                if len(entity_parts) == 2:
                    entity_type, entity_text = entity_parts

                    if (len(entity_text) >= min_entity_length and
                            self._is_valid_armenian_entity(entity_text) and
                            not self._looks_like_noise(entity_text)):
                        filtered_set.add(entity)

            filtered_entities.append(filtered_set)

        return filtered_entities

    def _looks_like_noise(self, text: str) -> bool:
        """Проверка, выглядит ли текст как случайный шум"""
        if re.search(r'(.)\1{2,}', text):  # 3 или более повторяющихся символов
            return True

        if re.search(r'^[ա-ֆ]{1,3}$', text) and len(text) < 3:
            return True

        return False