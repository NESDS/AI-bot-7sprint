"""
Скрипт для структурированного парсинга статей с fandom.com про Союзмультфильм.

Функциональность:
1. Скачивает страницы из заранее заданного словаря URL-адресов.
2. Извлекает заголовок, вводный текст, структуру разделов и данные инфобокса.
3. Сохраняет результат в Markdown-файлы (по одному файлу на источник) в папке raw_sources/.

Зависимости: requests, beautifulsoup4
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup, Tag

# --- Константы проекта
PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "raw_sources"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.0  # секунды между запросами, чтобы не спамить сервер

URLS = {
    # --- Простоквашино ---
    "film_troe_iz_prostokvashino.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A2%D1%80%D0%BE%D0%B5_%D0%B8%D0%B7_%D0%9F%D1%80%D0%BE%D1%81%D1%82%D0%BE%D0%BA%D0%B2%D0%B0%D1%88%D0%B8%D0%BD%D0%BE_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "film_kanikuly_v_prostokvashino.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9A%D0%B0%D0%BD%D0%B8%D0%BA%D1%83%D0%BB%D1%8B_%D0%B2_%D0%9F%D1%80%D0%BE%D1%81%D1%82%D0%BE%D0%BA%D0%B2%D0%B0%D1%88%D0%B8%D0%BD%D0%BE_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "film_zima_v_prostokvashino.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%97%D0%B8%D0%BC%D0%B0_%D0%B2_%D0%9F%D1%80%D0%BE%D1%81%D1%82%D0%BE%D0%BA%D0%B2%D0%B0%D1%88%D0%B8%D0%BD%D0%BE_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "series_prostokvashino.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9F%D1%80%D0%BE%D1%81%D1%82%D0%BE%D0%BA%D0%B2%D0%B0%D1%88%D0%B8%D0%BD%D0%BE",
    "person_dyadya_fedor.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%94%D1%8F%D0%B4%D1%8F_%D0%A4%D1%91%D0%B4%D0%BE%D1%80",
    "person_matroskin.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9C%D0%B0%D1%82%D1%80%D0%BE%D1%81%D0%BA%D0%B8%D0%BD",
    "person_sharik.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A8%D0%B0%D1%80%D0%B8%D0%BA",
    "person_pechkin.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9F%D0%BE%D1%87%D1%82%D0%B0%D0%BB%D1%8C%D0%BE%D0%BD_%D0%9F%D0%B5%D1%87%D0%BA%D0%B8%D0%BD",
    # --- Чебурашка ---
    "film_krokodil_gena.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9A%D1%80%D0%BE%D0%BA%D0%BE%D0%B4%D0%B8%D0%BB_%D0%93%D0%B5%D0%BD%D0%B0_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "film_cheburashka.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A7%D0%B5%D0%B1%D1%83%D1%80%D0%B0%D1%88%D0%BA%D0%B0_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "film_shapoklyak.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A8%D0%B0%D0%BF%D0%BE%D0%BA%D0%BB%D1%8F%D0%BA_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "film_cheburashka_idet_v_shkolu.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A7%D0%B5%D0%B1%D1%83%D1%80%D0%B0%D1%88%D0%BA%D0%B0_%D0%B8%D0%B4%D1%91%D1%82_%D0%B2_%D1%88%D0%BA%D0%BE%D0%BB%D1%83",
    "person_cheburashka.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A7%D0%B5%D0%B1%D1%83%D1%80%D0%B0%D1%88%D0%BA%D0%B0",
    "person_krokodil_gena_char.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9A%D1%80%D0%BE%D0%BA%D0%BE%D0%B4%D0%B8%D0%BB_%D0%93%D0%B5%D0%BD%D0%B0",
    "person_shapoklyak_char.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A1%D1%82%D0%B0%D1%80%D1%83%D1%85%D0%B0_%D0%A8%D0%B0%D0%BF%D0%BE%D0%BA%D0%BB%D1%8F%D0%BA",
    # --- Винни-Пух ---
    "film_vinni_puh.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%92%D0%B8%D0%BD%D0%BD%D0%B8-%D0%9F%D1%83%D1%85_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "film_vinni_puh_idet_v_gosti.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%92%D0%B8%D0%BD%D0%BD%D0%B8-%D0%9F%D1%83%D1%85_%D0%B8%D0%B4%D1%91%D1%82_%D0%B2_%D0%B3%D0%BE%D1%81%D1%82%D0%B8",
    "film_vinni_puh_i_den_zabot.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%92%D0%B8%D0%BD%D0%BD%D0%B8-%D0%9F%D1%83%D1%85_%D0%B8_%D0%B4%D0%B5%D0%BD%D1%8C_%D0%B7%D0%B0%D0%B1%D0%BE%D1%82",
    "person_vinni_puh.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%92%D0%B8%D0%BD%D0%BD%D0%B8-%D0%9F%D1%83%D1%85",
    "person_pyatachok.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9F%D1%8F%D1%82%D0%B0%D1%87%D0%BE%D0%BA",
    # --- Ну, погоди! ---
    "series_nu_pogodi.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9D%D1%83%2C_%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D0%B8%21",
    "series_nu_pogodi_kanikuly.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9D%D1%83%2C_%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D0%B8%21_%D0%9A%D0%B0%D0%BD%D0%B8%D0%BA%D1%83%D0%BB%D1%8B",
    "duo_zayats_i_volk.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%97%D0%B0%D1%8F%D1%86_%D0%B8_%D0%92%D0%BE%D0%BB%D0%BA",
    # --- Бременские музыканты ---
    "film_bremenskie_muzykanty.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%91%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D1%81%D0%BA%D0%B8%D0%B5_%D0%BC%D1%83%D0%B7%D1%8B%D0%BA%D0%B0%D0%BD%D1%82%D1%8B",
    "film_po_sledam_bremenskih.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9F%D0%BE_%D1%81%D0%BB%D0%B5%D0%B4%D0%B0%D0%BC_%D0%B1%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D1%81%D0%BA%D0%B8%D1%85_%D0%BC%D1%83%D0%B7%D1%8B%D0%BA%D0%B0%D0%BD%D1%82%D0%BE%D0%B2_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "group_bremenskie_muzykanty.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%91%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D1%81%D0%BA%D0%B8%D0%B5_%D0%9C%D1%83%D0%B7%D1%8B%D0%BA%D0%B0%D0%BD%D1%82%D1%8B_%28%D0%93%D1%80%D1%83%D0%BF%D0%BF%D0%B0%29",
    "person_trubadur.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A2%D1%80%D1%83%D0%B1%D0%B0%D0%B4%D1%83%D1%80",
    # --- Другая классика ---
    "film_zhil_byl_pes.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%96%D0%B8%D0%BB-%D0%B1%D1%8B%D0%BB_%D0%BF%D1%91%D1%81",
    "person_pes_zhil_byl_pes.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9F%D1%91%D1%81_%D0%B8%D0%B7_%C2%AB%D0%96%D0%B8%D0%BB-%D0%B1%D1%8B%D0%BB_%D0%BF%D1%91%D1%81%C2%BB",
    "film_ded_moroz_i_leto.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%94%D0%B5%D0%B4_%D0%9C%D0%BE%D1%80%D0%BE%D0%B7_%D0%B8_%D0%BB%D0%B5%D1%82%D0%BE_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
    "person_kotenok_gav.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%9A%D0%BE%D1%82%D1%91%D0%BD%D0%BE%D0%BA_%D0%93%D0%B0%D0%B2",
    "person_bonifaciy.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%91%D0%BE%D0%BD%D0%B8%D1%84%D0%B0%D1%86%D0%B8%D0%B9",
    "film_malchish_kibalchish.md": "https://soyuzmultfilm.fandom.com/ru/wiki/%D0%A1%D0%BA%D0%B0%D0%B7%D0%BA%D0%B0_%D0%BE_%D0%9C%D0%B0%D0%BB%D1%8C%D1%87%D0%B8%D1%88%D0%B5-%D0%9A%D0%B8%D0%B1%D0%B0%D0%BB%D1%8C%D1%87%D0%B8%D1%88%D0%B5_%28%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC%29",
}


@dataclass
class Section:
    title: str
    content: List[str] = field(default_factory=list)


def ensure_raw_dir() -> None:
    """Гарантирует, что папка raw_sources существует."""
    RAW_DIR.mkdir(exist_ok=True)


def fetch_html(url: str) -> str:
    """Скачивает HTML-страницу с указанием юзер-агента и таймаутом."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def clean_heading_text(raw_heading: str) -> str:
    """Удаляет лишние артефакты вроде [править] из заголовков."""
    cleaned = re.sub(r"\[.*?\]", "", raw_heading)
    return cleaned.strip()


def parse_infobox(soup: BeautifulSoup) -> List[str]:
    """Достаёт ключевые пары «поле-значение» из portable-infobox."""
    infobox = soup.select_one(".portable-infobox")
    if not infobox:
        return []

    lines: List[str] = []
    for item in infobox.select(".pi-item"):
        label = item.select_one(".pi-data-label")
        value = item.select_one(".pi-data-value")
        if not label or not value:
            continue
        label_text = " ".join(label.stripped_strings)
        value_text = " ".join(value.stripped_strings)
        if label_text and value_text:
            lines.append(f"- **{label_text}:** {value_text}")
    return lines


def iter_main_nodes(container: Tag) -> Iterable[Tag]:
    """Возвращает значимые узлы основного контента."""
    for node in container.children:
        if not isinstance(node, Tag):
            continue
        if node.name in {"script", "style", "table", "figure", "div"} and not node.get_text(strip=True):
            continue
        if "toc" in node.get("class", []):
            continue
        yield node


def node_to_markdown(node: Tag) -> str | None:
    """Конвертирует допустимый HTML-узел в строку Markdown."""
    if node.name == "p":
        text = node.get_text(" ", strip=True)
        return text or None
    if node.name == "ul":
        items = [
            f"- {li.get_text(' ', strip=True)}"
            for li in node.find_all("li", recursive=False)
            if li.get_text(strip=True)
        ]
        return "\n".join(items) or None
    if node.name == "ol":
        items = [
            f"{idx}. {li.get_text(' ', strip=True)}"
            for idx, li in enumerate(
                (li for li in node.find_all("li", recursive=False) if li.get_text(strip=True)),
                start=1,
            )
        ]
        return "\n".join(items) or None
    if node.name == "blockquote":
        text = node.get_text(" ", strip=True)
        return f"> {text}" if text else None

    text = node.get_text(" ", strip=True)
    return text or None


def extract_sections(main: Tag) -> List[Section]:
    """Формирует список разделов с содержимым."""
    sections: List[Section] = []
    current = Section(title="Введение")

    for node in iter_main_nodes(main):
        if node.name and node.name.startswith("h"):
            level = int(node.name[1])
            if level <= 3:  # ограничимся до h3
                cleaned_title = clean_heading_text(node.get_text(strip=True))
                if current.content:
                    sections.append(current)
                current = Section(title=cleaned_title or f"Section {len(sections) + 1}")
                continue

        markdown = node_to_markdown(node)
        if markdown:
            current.content.append(markdown)

    if current.content:
        sections.append(current)

    return sections


def render_markdown(title: str, url: str, infobox: List[str], sections: List[Section]) -> str:
    """Собирает Markdown-документ по результатам парсинга."""
    lines = [
        f"# {title}",
        "",
        f"**Источник:** {url}",
        "",
    ]

    if infobox:
        lines.append("## Сводка из инфобокса")
        lines.extend(infobox)
        lines.append("")

    for section in sections:
        lines.append(f"## {section.title}")
        lines.append("")
        for block in section.content:
            lines.append(block)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_page(url: str) -> str:
    """Скачивает и преобразует страницу в Markdown."""
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.select_one("#firstHeading")
    if not title_tag:
        raise ValueError("Не удалось извлечь заголовок страницы")
    title = title_tag.get_text(strip=True)

    main_content = soup.select_one(".mw-parser-output")
    if not main_content:
        raise ValueError("Не удалось найти основной контент страницы")

    infobox_lines = parse_infobox(soup)
    sections = extract_sections(main_content)
    return render_markdown(title, url, infobox_lines, sections)


def save_markdown(filename: str, content: str) -> None:
    """Сохраняет текст в файл внутри raw_sources/."""
    (RAW_DIR / filename).write_text(content, encoding="utf-8")


def main() -> None:
    ensure_raw_dir()
    for idx, (filename, url) in enumerate(URLS.items(), start=1):
        print(f"[{idx}/{len(URLS)}] Парсим {url}")
        try:
            md_text = parse_page(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! Ошибка при обработке {url}: {exc}")
            continue

        save_markdown(filename, md_text)
        print(f"  -> сохранено в {RAW_DIR / filename}")
        time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    main()

