"""
RAG-бот на основе ReAct агента (LangGraph) для Telegram
Использует векторную базу ChromaDB и GigaChat API
"""

import os
import logging
from dotenv import load_dotenv

# Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# LangChain & LangGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langgraph.prebuilt import create_react_agent

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальные переменные
vectorstore = None
llm = None
agent_executor = None

# Хранилище истории диалогов: {user_id: [messages]}
chat_histories = {}


# ============================================================================
# 1. ИНСТРУМЕНТ ДЛЯ ПОИСКА В ВЕКТОРНОЙ БАЗЕ
# ============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    Ищет информацию в базе знаний о вселенной СказКадр (мультфильмы).
    Используй этот инструмент, когда нужно найти факты о персонажах, сюжетах, фильмах.
    
    Args:
        query: поисковый запрос на русском языке
        
    Returns:
        Релевантные фрагменты из базы знаний
    """
    try:
        # Поиск топ-4 релевантных фрагментов
        results = vectorstore.similarity_search(query, k=4)
        
        if not results:
            return "Информация не найдена в базе знаний."
        
        # Форматирование результатов
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'неизвестно')
            title = doc.metadata.get('title', 'без названия')
            content = doc.page_content.strip()
            
            context_parts.append(
                f"[Источник {i}: {title}]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    except Exception as e:
        logger.error(f"Ошибка при поиске в базе знаний: {e}")
        return f"Ошибка при поиске: {str(e)}"


# ============================================================================
# 2. ПРОМПТЫ С FEW-SHOT И CHAIN-OF-THOUGHT
# ============================================================================

SYSTEM_PROMPT = """Ты — интеллектуальный помощник-эксперт по вселенной мультфильмов студии СказКадр.

**Твоя задача:**
1. Отвечать на вопросы о персонажах, сюжетах и фильмах вселенной СказКадр
2. Использовать инструмент поиска для получения точной информации из базы знаний
3. Применять Chain-of-Thought: сначала рассуждай, потом отвечай

**Правила рассуждения (Chain-of-Thought):**
- Всегда начинай с шага "Размышление:", где описываешь свой план
- Используй инструмент search_knowledge_base для поиска фактов
- После получения информации, анализируй её в шаге "Анализ:"
- Формулируй финальный ответ в шаге "Ответ:"

**Few-shot примеры:**

ПРИМЕР 1:
Вопрос: Кто такой кот Шнурок и где он живёт?
Размышление: Мне нужно найти информацию о коте Шнурке и месте его проживания. Использую поиск.
[Использован инструмент search_knowledge_base с запросом "кот Шнурок Лапушкино"]
Анализ: В найденных фрагментах указано, что кот Шнурок — это умный и хозяйственный кот, который живёт в деревне Лапушкино вместе со Степаном-молчуном и псом Шариком.
Ответ: Кот Шнурок — один из главных персонажей, живущий в деревне Лапушкино. Это умный и практичный кот, известный своим хозяйственным подходом к жизни.

ПРИМЕР 2:
Вопрос: Расскажи про Ушарика.
Размышление: Нужно найти информацию о персонаже по имени Ушарик. Ищу в базе знаний.
[Использован инструмент search_knowledge_base с запросом "Ушарик персонаж"]
Анализ: Ушарик — это загадочное существо с большими ушами, главный герой нескольких мультфильмов. Он дружит с крокодилом Гектором.
Ответ: Ушарик — милое существо с огромными ушами, которое стало символом доброты и дружбы. Его лучший друг — крокодил Гектор, вместе они переживают множество приключений.

**Если информации нет в базе знаний:**
Если поиск не дал результатов, честно скажи: "К сожалению, я не нашёл информации по этому вопросу в своей базе знаний о вселенной СказКадр."

**Важно:**
- НЕ придумывай факты, используй только информацию из базы знаний
- Всегда показывай своё рассуждение перед ответом
- Отвечай на русском языке, дружелюбно и структурированно
"""


# ============================================================================
# 3. REACT АГЕНТ НА LANGGRAPH
# ============================================================================

def create_agent():
    """Создаёт ReAct агента с помощью prebuilt функции LangGraph"""
    
    # Инструменты агента
    tools = [search_knowledge_base]
    
    # Создание ReAct агента с системным промптом
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT  # Системный промпт с Few-shot и CoT
    )
    
    return agent_executor


# ============================================================================
# 4. TELEGRAM BOT HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_message = """🎬 Привет! Я бот-эксперт по вселенной СказКадр!

Я знаю всё о мультфильмах, персонажах и сюжетах:
• Лапушкино и его обитатели (Шнурок, Степан-молчун, пёс Шарик)
• Ушарик и крокодил Гектор
• Пухнастик и Рыльчик
• И многие другие герои!

Просто задай мне любой вопрос о вселенной СказКадр! 🎭

Примеры вопросов:
- Кто такой кот Шнурок?
- Расскажи про деревню Лапушкино
- Кто друг Ушарика?
"""
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """📚 Как пользоваться ботом:

1. Просто напиши свой вопрос о вселенной СказКадр
2. Я найду информацию в базе знаний
3. Покажу своё рассуждение (Chain-of-Thought)
4. Дам точный ответ на основе фактов

Доступные команды:
/start - Начать работу с ботом
/help - Показать эту справку
/examples - Примеры вопросов
/clear - Очистить историю диалога

💬 Я помню весь наш разговор! Используй /clear чтобы начать с чистого листа.

❗ Если я не найду информацию, честно об этом скажу!
"""
    await update.message.reply_text(help_text)


async def examples_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /examples"""
    examples_text = """💡 Примеры вопросов:

🏘️ О локациях:
- Что такое Лапушкино?
- Где живёт кот Шнурок?

👥 О персонажах:
- Кто такой Ушарик?
- Расскажи про крокодила Гектора
- Кто такой Степан-молчун?
- Что ты знаешь про Пухнастика?

🎬 О сюжетах:
- О чём мультфильм про Лапушкино?
- Какие приключения были у Ушарика?

Задавай свои вопросы! 🎭
"""
    await update.message.reply_text(examples_text)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /clear - очистка истории диалога"""
    user_id = update.effective_user.id
    
    if user_id in chat_histories:
        del chat_histories[user_id]
        logger.info(f"История диалога очищена для пользователя {user_id}")
    
    clear_text = """🗑️ История диалога очищена!

Теперь я не помню наш предыдущий разговор.
Можешь начать новый диалог с чистого листа! 🎬
"""
    await update.message.reply_text(clear_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений (основная логика RAG)"""
    user_message = update.message.text
    user_id = update.effective_user.id
    
    logger.info(f"Получен вопрос от пользователя {user_id}: {user_message}")
    
    # Отправка "typing..." индикатора
    await update.message.chat.send_action(action="typing")
    
    try:
        # Получаем историю диалога для пользователя (или создаем новую)
        if user_id not in chat_histories:
            chat_histories[user_id] = []
        
        history = chat_histories[user_id]
        
        # Добавляем новый вопрос к истории
        history.append(HumanMessage(content=user_message))
        
        # Вызов ReAct агента с полной историей
        result = agent_executor.invoke({"messages": history})
        
        # Получаем все сообщения из результата (включая промежуточные шаги агента)
        result_messages = result.get("messages", [])
        
        # Извлечение финального ответа (последнее AIMessage без tool_calls)
        response_text = ""
        for msg in reversed(result_messages):
            if hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage":
                # Проверяем, есть ли tool_calls
                has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                if not has_tool_calls and hasattr(msg, "content") and msg.content:
                    response_text = msg.content
                    break
        
        if not response_text:
            response_text = "Извини, не смог сформировать ответ. Попробуй переформулировать вопрос."
        
        # Обновляем историю результатами работы агента
        chat_histories[user_id] = result_messages
        
        # Отправка ответа пользователю
        await update.message.reply_text(response_text)
        
        logger.info(f"Отправлен ответ пользователю {user_id}")
    
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        error_message = "😔 Извини, произошла ошибка при обработке твоего вопроса. Попробуй ещё раз!"
        await update.message.reply_text(error_message)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Update {update} caused error {context.error}")


# ============================================================================
# 5. ИНИЦИАЛИЗАЦИЯ И ЗАПУСК
# ============================================================================

def initialize_rag_system():
    """Инициализация RAG-системы: загрузка векторной базы и LLM"""
    global vectorstore, llm, agent_executor
    
    logger.info("Инициализация RAG-системы...")
    
    # Параметры GigaChat
    gigachat_auth = os.getenv("GIGACHAT_AUTH")
    gigachat_scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
    gigachat_model = os.getenv("GIGACHAT_MODEL", "GigaChat-Max")
    
    # Загрузка эмбеддинговой модели
    logger.info("Загрузка эмбеддинговой модели GigaChat...")
    embeddings = GigaChatEmbeddings(
        credentials=gigachat_auth,
        scope=gigachat_scope,
        verify_ssl_certs=False
    )
    
    # Загрузка векторной базы ChromaDB
    logger.info("Загрузка векторной базы ChromaDB...")
    persist_directory = "artifacts/chroma_index"
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
        # collection_name не указываем - используем дефолтное имя, как при создании индекса
    )
    
    logger.info(f"Векторная база загружена. Коллекция: {vectorstore._collection.count()} документов")
    
    # Инициализация LLM
    logger.info("Инициализация GigaChat LLM...")
    llm = GigaChat(
        credentials=gigachat_auth,
        scope=gigachat_scope,
        model=gigachat_model,
        verify_ssl_certs=False,
        timeout=120,
        temperature=0.1  # Низкая температура для более точных ответов
    )
    
    # Создание ReAct агента
    logger.info("Создание ReAct агента на LangGraph...")
    agent_executor = create_agent()
    
    logger.info("✅ RAG-система успешно инициализирована!")


def main():
    """Основная функция запуска бота"""
    
    # Инициализация RAG-системы
    initialize_rag_system()
    
    # Получение токена Telegram
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env файле!")
    
    # Создание приложения
    logger.info("Запуск Telegram бота...")
    application = Application.builder().token(telegram_token).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("examples", examples_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    # Запуск бота
    logger.info("🚀 Бот запущен и готов к работе!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

