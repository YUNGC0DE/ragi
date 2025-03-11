#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для создания кастомного RAG с использованием:
1. Локальной модели эмбеддингов jinaai/jina-embeddings-v3
2. Google модели для генерации
3. Текстов из директории train_texts
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from ragbuilder import RAGBuilder
from ragbuilder.config import DataIngestOptionsConfig
from ragbuilder.config.components import ChunkingStrategy, EmbeddingType, ParserType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Загрузка переменных окружения (если есть)
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_google_api_key():
    """Проверка и настройка Google API ключа"""
    if "GOOGLE_API_KEY" not in os.environ:
        api_key = input("Введите ваш Google API ключ: ")
        os.environ["GOOGLE_API_KEY"] = api_key
    return os.environ["GOOGLE_API_KEY"]

def main():
    # Настройка Google API ключа
    setup_google_api_key()
    
    # Путь к директории с текстами
    input_dir = Path("train_texts")
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Директория {input_dir} не существует или не является директорией")
    
    # Создание модели эмбеддингов (локальная модель jinaai/jina-embeddings-v3)
    logger.info("Инициализация модели эмбеддингов jinaai/jina-embeddings-v3...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Создание модели для генерации (Google)
    logger.info("Инициализация модели Google для генерации...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        convert_system_message_to_human=True
    )
    
    # Создание конфигурации для обработки данных
    logger.info("Создание конфигурации для обработки данных...")
    data_ingest_config = DataIngestOptionsConfig(
        input_source=str(input_dir),
        document_loaders=[
            {"type": ParserType.UNSTRUCTURED}  # Хорошо работает с TXT файлами
        ],
        chunking_strategies=[
            # Стандартный рекурсивный разделитель
            {"type": ChunkingStrategy.RECURSIVE},
            # Контекстуальный разделитель
            {"type": ChunkingStrategy.CUSTOM, 
             "custom_class": "ragbuilder.custom_components.ContextualChunker"}
        ],
        # Диапазон размеров чанков для оптимизации
        chunk_size={"min": 500, "max": 1500, "stepsize": 250},
        # Диапазон перекрытий
        chunk_overlap={"min": 50, "max": 200, "stepsize": 50},
        # Модели эмбеддингов для оптимизации
        embedding_models=[
            {"type": EmbeddingType.HUGGINGFACE, 
             "model_kwargs": {"model_name": "jinaai/jina-embeddings-v3"}}
        ],
        # Векторные базы данных
        vector_databases=[
            {"type": "chroma", "collection_name": "custom_rag_collection"},
            {"type": "faiss"}
        ],
        # Количество документов для извлечения
        top_k=[3, 5, 10]
    )
    
    # Создание RAGBuilder с нашей конфигурацией
    logger.info("Создание RAGBuilder...")
    builder = RAGBuilder(
        data_ingest_config=data_ingest_config,
        default_llm=llm,
        default_embeddings=embeddings,
        n_trials=10  # Количество попыток оптимизации
    )
    
    # Запуск оптимизации
    logger.info("Запуск оптимизации...")
    results = builder.optimize()
    
    # Сохранение оптимизированной конфигурации
    logger.info("Сохранение оптимизированной конфигурации...")
    builder.save("optimized_rag_config")
    
    # Вывод результатов
    logger.info("Оптимизация завершена!")
    logger.info(f"Лучшая конфигурация: {results.data_ingest.best_config}")
    
    # Тестирование RAG
    test_question = "Расскажи о ключевых моментах из этих документов"
    logger.info(f"Тестовый вопрос: {test_question}")
    answer = builder.query(test_question)
    logger.info(f"Ответ: {answer}")
    
    # Запуск API-сервера
    logger.info("Запуск API-сервера...")
    builder.serve()

if __name__ == "__main__":
    main() 