"""
Скрипт для применения замены терминов из terms_map.json к документам.

Берёт все .md файлы из raw_sources/, применяет замены согласно словарю terms_map.json
и сохраняет результаты в knowledge_base/.

Замены применяются контекстно: для каждого файла используется только релевантная группа терминов.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

# --- Константы проекта
PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "raw_sources"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
TERMS_MAP_FILE = PROJECT_ROOT / "terms_map.json"

# Маппинг префиксов файлов на группы терминов
FILE_TO_GROUPS = {
    # Простоквашино
    "film_troe_iz_prostokvashino": ["common", "prostokvashino"],
    "film_kanikuly_v_prostokvashino": ["common", "prostokvashino"],
    "film_zima_v_prostokvashino": ["common", "prostokvashino"],
    "series_prostokvashino": ["common", "prostokvashino"],
    "person_dyadya_fedor": ["common", "prostokvashino"],
    "person_matroskin": ["common", "prostokvashino"],
    "person_sharik": ["common", "prostokvashino"],
    "person_pechkin": ["common", "prostokvashino"],
    
    # Чебурашка
    "film_krokodil_gena": ["common", "cheburashka"],
    "film_cheburashka": ["common", "cheburashka"],
    "film_shapoklyak": ["common", "cheburashka"],
    "film_cheburashka_idet_v_shkolu": ["common", "cheburashka"],
    "person_cheburashka": ["common", "cheburashka"],
    "person_krokodil_gena_char": ["common", "cheburashka"],
    "person_shapoklyak_char": ["common", "cheburashka"],
    
    # Винни-Пух
    "film_vinni_puh": ["common", "vinni_puh"],
    "film_vinni_puh_idet_v_gosti": ["common", "vinni_puh"],
    "film_vinni_puh_i_den_zabot": ["common", "vinni_puh"],
    "person_vinni_puh": ["common", "vinni_puh"],
    "person_pyatachok": ["common", "vinni_puh"],
    
    # Ну, погоди!
    "series_nu_pogodi": ["common", "nu_pogodi"],
    "series_nu_pogodi_kanikuly": ["common", "nu_pogodi"],
    "duo_zayats_i_volk": ["common", "nu_pogodi"],
    
    # Бременские музыканты
    "film_bremenskie_muzykanty": ["common", "bremenskie"],
    "film_po_sledam_bremenskih": ["common", "bremenskie"],
    "group_bremenskie_muzykanty": ["common", "bremenskie"],
    "person_trubadur": ["common", "bremenskie"],
    
    # Жил-был пёс
    "film_zhil_byl_pes": ["common", "zhil_byl_pes"],
    "person_pes_zhil_byl_pes": ["common", "zhil_byl_pes"],
    
    # Дед Мороз и лето
    "film_ded_moroz_i_leto": ["common", "ded_moroz"],
    
    # Котёнок Гав
    "person_kotenok_gav": ["common", "kotenok_gav"],
    
    # Бонифаций
    "person_bonifaciy": ["common", "bonifaciy"],
    
    # Мальчиш-Кибальчиш
    "film_malchish_kibalchish": ["common", "malchish_kibalchish"],
}


def load_terms_map() -> Dict[str, Dict[str, str]]:
    """Загружает словарь замен из terms_map.json."""
    with open(TERMS_MAP_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Убираем комментарий
    data.pop("_comment", None)
    return data


def get_terms_for_file(filename: str, all_terms: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Определяет, какие группы терминов нужно применить для данного файла.
    
    Возвращает объединённый словарь релевантных терминов.
    """
    # Убираем расширение .md
    file_stem = filename.replace(".md", "")
    
    # Определяем группы для этого файла
    groups = FILE_TO_GROUPS.get(file_stem, ["common"])
    
    # Объединяем термины из всех релевантных групп
    result = {}
    for group_name in groups:
        if group_name in all_terms:
            result.update(all_terms[group_name])
    
    return result


def sort_terms_by_length(terms_dict: Dict[str, str]) -> list[tuple[str, str]]:
    """
    Сортирует термины по длине (от длинных к коротким).
    
    Это важно для правильного порядка замен:
    - Сначала заменяем "кот Матроскин" -> "кот Шнурок"
    - Потом "Матроскин" -> "кот Шнурок"
    - И только в конце "Кот" -> "Усач" (из Бременских музыкантов)
    """
    return sorted(terms_dict.items(), key=lambda x: len(x[0]), reverse=True)


def replace_terms(text: str, terms_map: Dict[str, str]) -> str:
    """
    Заменяет все термины в тексте согласно словарю.
    
    Использует границы слов (\b) для точной замены, но учитывает
    кириллические символы и дефисы в составных словах.
    """
    result = text
    sorted_terms = sort_terms_by_length(terms_map)
    
    for old_term, new_term in sorted_terms:
        # Экранируем специальные символы регулярных выражений
        escaped_term = re.escape(old_term)
        
        # Заменяем термин (с учётом границ слова)
        pattern = rf'\b{escaped_term}\b'
        result = re.sub(pattern, new_term, result)
    
    return result


def process_file(source_path: Path, target_path: Path, all_terms: Dict[str, Dict[str, str]]) -> None:
    """Обрабатывает один файл: читает, заменяет термины, сохраняет."""
    # Читаем исходный файл
    with open(source_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Получаем релевантные термины для этого файла
    file_terms = get_terms_for_file(source_path.name, all_terms)
    
    # Применяем замены
    new_content = replace_terms(content, file_terms)
    
    # Сохраняем в новый файл
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    # Выводим информацию о применённых группах
    file_stem = source_path.name.replace(".md", "")
    groups = FILE_TO_GROUPS.get(file_stem, ["common"])
    print(f"OK {source_path.name:40} -> [{', '.join(groups)}] ({len(file_terms)} терминов)")


def main():
    """Основная функция: обрабатывает все файлы из raw_sources/."""
    print("=" * 80)
    print("Применение замен терминов из terms_map.json (контекстный режим)")
    print("=" * 80)
    
    # Загружаем словарь замен
    print(f"\n[1/3] Загрузка словаря замен из {TERMS_MAP_FILE.name}...")
    all_terms = load_terms_map()
    total_terms = sum(len(terms) for terms in all_terms.values())
    print(f"      Загружено {len(all_terms)} групп, всего {total_terms} терминов")
    
    # Получаем список всех .md файлов в raw_sources
    print(f"\n[2/3] Поиск файлов в {RAW_DIR.name}/...")
    md_files = sorted(RAW_DIR.glob("*.md"))
    print(f"      Найдено {len(md_files)} файлов")
    
    # Обрабатываем каждый файл
    print(f"\n[3/3] Обработка файлов и сохранение в {KNOWLEDGE_BASE_DIR.name}/...\n")
    KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
    
    for i, source_file in enumerate(md_files, 1):
        target_file = KNOWLEDGE_BASE_DIR / source_file.name
        print(f"[{i:2}/{len(md_files)}] ", end="")
        process_file(source_file, target_file, all_terms)
    
    print("\n" + "=" * 80)
    print(f"[OK] Готово! Обработано файлов: {len(md_files)}")
    print(f"[OK] Результаты сохранены в: {KNOWLEDGE_BASE_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
