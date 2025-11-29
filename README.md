# 🎬 RAG-бот СказКадр

Telegram-бот с RAG-архитектурой для ответов на вопросы о вымышленной вселенной мультфильмов.

## 🛠 Технологии

- **LLM**: GigaChat-2-Max
- **Эмбеддинги**: GigaChatEmbeddings
- **Векторная БД**: ChromaDB
- **Агент**: LangGraph ReAct
- **Интерфейс**: Telegram Bot

## 📋 Структура проекта

```
├── rag_bot.py              # Telegram-бот с ReAct агентом
├── build_index.py          # Создание векторного индекса
├── parse_sources.py        # Парсинг источников
├── apply_terms_replacement.py  # Замена терминов
├── terms_map.json          # Словарь замен
├── knowledge_base/         # База знаний (34 документа)
├── artifacts/chroma_index/ # Векторный индекс
├── screenshots/            # Скриншоты тестирования
└── Project_template.md     # Полная документация
```

## 🎯 Выполненные задания

### Задание 1. Исследование
Сравнение LLM-моделей, эмбеддингов и векторных БД. Выбор стека: GigaChat + ChromaDB.

### Задание 2. База знаний
Парсинг 33 страниц Союзмультфильма → замена терминов на вымышленные (Простоквашино → Лапушкино, Чебурашка → Ушарик).

### Задание 3. Векторный индекс
ChromaDB с 125 чанками (1000 символов, overlap 150). GigaChatEmbeddings для векторизации.

### Задание 4. RAG-бот
- ReAct агент на LangGraph
- Few-shot prompting (3 примера)
- Chain-of-Thought рассуждения
- Память диалога
- Команды: `/start`, `/help`, `/examples`, `/mode`, `/clear`

### Задание 5. Безопасность
Трёхуровневая защита от промпт-инъекций:
1. Фильтрация вредоносных чанков (20+ паттернов)
2. Правила безопасности в промпте
3. Встроенные safety-фильтры GigaChat

**Вывод**: Даже при отключённой пользовательской защите, GigaChat отказывается выводить пароли — защита на уровне провайдера.

## 🚀 Запуск

```bash
# Установка зависимостей
uv pip install -r requirements.txt

# Создание индекса
python build_index.py

# Запуск бота
python rag_bot.py
```

## ⚙️ Конфигурация

Создайте `.env` файл:
```
GIGACHAT_AUTH=your_auth_token
GIGACHAT_SCOPE=GIGACHAT_API_CORP
GIGACHAT_MODEL=GigaChat-2-Max
TELEGRAM_BOT_TOKEN=your_bot_token
```

## 📸 Демонстрация

Скриншоты тестирования находятся в папке `screenshots/` (12 файлов).

Подробная документация: [Project_template.md](Project_template.md)

