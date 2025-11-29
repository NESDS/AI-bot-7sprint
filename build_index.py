"""
Скрипт создания векторного индекса с использованием Chroma и GigaChat Embeddings.

Последовательная обработка: загрузка документов, разбиение на чанки, построение индекса.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import typer
from dotenv import load_dotenv
from langchain_community.embeddings import GigaChatEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


DEFAULT_PROBE_QUERIES = [
    "Кто такой кот Шнурок и чем он знаменит в Лапушкино?",
    "О чём сюжет фильма про Ушарика и Шапокляк?",
]


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Отсутствует обязательная переменная окружения: {name}")
    return value


@dataclass
class PipelineConfig:
    """Настройки пайплайна индексации."""

    knowledge_base_dir: Path
    persist_directory: Path
    chunk_size: int = 1000
    chunk_overlap: int = 150
    sample_queries: List[str] = field(default_factory=lambda: DEFAULT_PROBE_QUERIES)

    # Параметры GigaChat
    gigachat_auth: str = field(default_factory=lambda: _require_env("GIGACHAT_AUTH"))
    gigachat_scope: str = field(default_factory=lambda: _require_env("GIGACHAT_SCOPE"))
    gigachat_model: str = field(default_factory=lambda: _require_env("GIGACHAT_MODEL"))
    gigachat_api_url: str = field(default_factory=lambda: _require_env("GIGACHAT_API_URL"))
    gigachat_token_url: str = field(default_factory=lambda: _require_env("GIGACHAT_TOKEN_URL"))
    gigachat_embedding_url: str = field(default_factory=lambda: _require_env("GIGACHAT_EMBEDDING_URL"))

    def validate(self) -> None:
        if not self.knowledge_base_dir.exists():
            raise FileNotFoundError(f"Каталог базы знаний не найден: {self.knowledge_base_dir}")
        # Создаем родительскую директорию, если не существует
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)


def load_documents(config: PipelineConfig) -> List[Document]:
    """Загружает markdown-файлы из директории базы знаний и преобразует их в объекты Document."""
    documents: List[Document] = []
    for path in sorted(config.knowledge_base_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = _extract_title(text, default=path.stem)
        metadata = {
            "source": str(path.relative_to(config.knowledge_base_dir.parent)),
            "document_id": path.stem,
            "title": title,
        }
        documents.append(Document(page_content=text, metadata=metadata))
    logging.info("Загружено %s документов из %s", len(documents), config.knowledge_base_dir)
    return documents


def _extract_title(text: str, default: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip() or default
    return default


def split_documents(documents: List[Document], config: PipelineConfig) -> List[Document]:
    """Разбивает документы на фрагменты заданного размера с перекрытием."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['document_id']}::chunk_{idx}"
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    logging.info("Создано %s фрагментов с размером %s символов", len(chunks), config.chunk_size)
    return chunks


def clean_vectorstore_directory(directory: Path) -> None:
    """Удаляет директорию с существующим векторным хранилищем."""
    if directory.exists():
        shutil.rmtree(directory)
        logging.info("Удалена существующая директория: %s", directory)


def build_vectorstore(chunks: List[Document], config: PipelineConfig) -> Chroma:
    """Создает векторное хранилище Chroma на основе фрагментов документов."""
    # Удаляем старую базу перед созданием новой
    clean_vectorstore_directory(config.persist_directory)
    
    embeddings = GigaChatEmbeddings(
        credentials=config.gigachat_auth,
        scope=config.gigachat_scope,
        verify_ssl_certs=False,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.persist_directory),
    )
    vectorstore.persist()
    logging.info("Векторное хранилище сохранено в %s", config.persist_directory)
    return vectorstore


def run_quality_probe(vectorstore: Chroma, queries: List[str], k: int = 3) -> Dict[str, List[Tuple[str, str]]]:
    """Выполняет тестовые запросы к индексу и возвращает результаты поиска."""
    probe_results: Dict[str, List[Tuple[str, str]]] = {}
    for query in queries:
        hits = vectorstore.similarity_search(query, k=k)
        probe_results[query] = [
            (hit.metadata.get("title", "no-title"), hit.page_content[:220].strip()) for hit in hits
        ]
    return probe_results


def save_stats(
    config: PipelineConfig,
    documents: List[Document],
    chunks: List[Document],
    stats_path: Path | None = None,
) -> Path:
    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "persist_directory": str(config.persist_directory),
    }
    stats_file = stats_path or config.persist_directory / "index_stats.json"
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats_file


def cli(
    knowledge_base: Path = typer.Option(
        Path("knowledge_base"), help="Каталог с подготовленной базой знаний."
    ),
    persist_dir: Path = typer.Option(
        Path("artifacts/chroma_index"), help="Куда сохранить Chroma-базу."
    ),
    chunk_size: int = typer.Option(1000, help="Размер чанка."),
    chunk_overlap: int = typer.Option(150, help="Перекрытие чанков."),
    probe: bool = typer.Option(True, help="Запустить тестовые запросы после индексации."),
) -> None:
    load_dotenv()

    config = PipelineConfig(
        knowledge_base_dir=knowledge_base,
        persist_directory=persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    config.validate()

    # Последовательное выполнение этапов индексации
    logging.info("Шаг 1/3: Загрузка документов...")
    raw_docs = load_documents(config)
    
    logging.info("Шаг 2/3: Разбиение на чанки...")
    chunk_docs = split_documents(raw_docs, config)
    
    logging.info("Шаг 3/3: Создание индекса...")
    vectorstore = build_vectorstore(chunk_docs, config)

    stats_path = save_stats(config, raw_docs, chunk_docs)
    logging.info("Статистика сохранена в %s", stats_path)

    if probe:
        probe_results = run_quality_probe(vectorstore, config.sample_queries)
        for query, hits in probe_results.items():
            logging.info("Запрос: %s", query)
            for title, snippet in hits:
                logging.info(" → %s: %s...", title, snippet.replace("\n", " "))

    typer.echo("Индексация завершена. Векторное хранилище готово к использованию.")


if __name__ == "__main__":
    typer.run(cli)

