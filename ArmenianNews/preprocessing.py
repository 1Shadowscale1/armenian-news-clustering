import pandas as pd
import re
from typing import Optional, Union
from datetime import datetime

class ArmenianTextPreprocessor:
    """Препроцессинг текстов на армянском языке"""

    def __init__(self):
        # Словарь для перевода армянских месяцев
        self.armenian_months = {
            'Հունվար': 'January', 'Փետրվար': 'February', 'Մարտ': 'March',
            'Ապրիլ': 'April', 'Մայիս': 'May', 'Հունիս': 'June',
            'Հուլիս': 'July', 'Օգոստոս': 'August', 'Սեպտեմբեր': 'September',
            'Հոկտեմբեր': 'October', 'Նոյեմբեր': 'November', 'Դեկտեմբեր': 'December',
            'հունվարի': 'January', 'փետրվարի': 'February', 'մարտի': 'March',
            'ապրիլի': 'April', 'մայիսի': 'May', 'հունիսի': 'June',
            'հուլիսի': 'July', 'օգոստոսի': 'August', 'սեպտեմբերի': 'September',
            'հոկտեմբերի': 'October', 'նոյեմբերի': 'November', 'դեկտեմբերի': 'December'
        }

    def convert_armenian_date(self, date_string: str) -> Optional[pd.Timestamp]:
        """Конвертирует армянскую дату в стандартный datetime формат"""
        if pd.isna(date_string):
            return pd.NaT

        try:
            # Заменяем армянские названия месяцев на английские
            date_clean = str(date_string).strip()
            for arm_month, eng_month in self.armenian_months.items():
                date_clean = date_clean.replace(arm_month, eng_month)

            # Удаляем лишние пробелы
            date_clean = ' '.join(date_clean.split())

            # Пробуем разные форматы дат
            formats_to_try = [
                '%d %B %Y %H:%M',    # "10 January 2024 14:30"
                '%d %B %Y',           # "10 January 2024"
                '%Y-%m-%d %H:%M:%S',  # "2024-01-10 14:30:00"
                '%Y-%m-%d',           # "2024-01-10"
                '%d/%m/%Y %H:%M',     # "10/01/2024 14:30"
                '%d.%m.%Y %H:%M'      # "10.01.2024 14:30"
            ]

            for fmt in formats_to_try:
                try:
                    return pd.to_datetime(date_clean, format=fmt)
                except:
                    continue

            # Если ни один формат не подошел, пробуем автоматическое определение
            return pd.to_datetime(date_clean, errors='coerce')

        except Exception as e:
            print(f"⚠️ Could not parse date: '{date_string}' - {e}")
            return pd.NaT

    def preprocess_dataframe(self, df: pd.DataFrame, date_column: str = 'date_time') -> pd.DataFrame:
        """Препроцессинг всего DataFrame"""
        # Конвертация дат
        df[date_column] = df[date_column].apply(self.convert_armenian_date)

        # Удаление строк с пустыми датами
        df = df.dropna(subset=[date_column]).reset_index(drop=True)

        # Создание объединенного текста
        df['full_text'] = df.apply(
            lambda row: f"{row['title']}. {row['text']}"[:1000],  # Обрезка длинных текстов
            axis=1
        )

        return df