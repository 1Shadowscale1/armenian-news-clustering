import re
from typing import Set, List


def normalize_armenian_entity(entity_text: str) -> str:
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


def enhanced_armenian_stemmer(word: str) -> str:
    """Улучшенный стеммер для армянского языка"""
    if not is_valid_armenian_entity(word):
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
    if not is_valid_armenian_entity(word):
        return ""

    return word


def is_valid_armenian_entity(entity_text: str) -> bool:
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
        r'^[^ա-ֆ]*$',  # Нет армянских букв
    ]

    for pattern in noise_patterns:
        if re.match(pattern, entity_text):
            return False

    # Фильтрация стоп-слов и шума
    armenian_stop_words = {
        'ը', 'ն', 'ում', 'իս', 'ով', 'ից', 'ան', 'ուն', 'յուն',
        'ություն', 'աց', 'եց', 'իկ', 'ակ', 'ային', 'ու', 'ի', 'է',
        'ե', 'ա', 'ո', 'չ', 'պ', 'հ', 'կ', 'մ', 'վ', 'տ', 'ր', 'ս',
        'դ', 'գ', 'լ', 'զ', 'ղ', 'ճ', 'ք', 'ֆ', 'ձ', 'ջ', 'ն', 'բ',
        'ըն', 'անց', 'աց', 'եց', 'ված', 'վածք', 'ում', 'ույն'
    }

    if entity_text in armenian_stop_words:
        return False

    # Проверка на значимые армянские слова
    if len(set(entity_text)) == 1 and len(entity_text) > 1:
        return False

    return True


def entity_similarity(ents1: Set[str], ents2: Set[str]) -> float:
    """Jaccard-сходство между множествами сущностей"""

    # Объединяем все непустые множества из каждого списка
    union_ents1 = set().union(*[ent for ent in ents1 if ent])
    union_ents2 = set().union(*[ent for ent in ents2 if ent])

    if not union_ents1 or not union_ents2:
        return 0

    intersection = len(union_ents1 & union_ents2)
    union_total = len(union_ents1 | union_ents2)

    return intersection / union_total if union_total > 0 else 0