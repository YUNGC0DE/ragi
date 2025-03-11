#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Основной скрипт для запуска полного процесса:
1. Генерация синтетических данных (вопросов и ответов)
2. Настройка и оптимизация RAG с использованием локальной модели эмбеддингов
"""

import os
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения (если есть)
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Проверка и установка необходимых зависимостей"""
    required_packages = [
        "langchain-huggingface",
        "langchain-google-genai",
        "sentence-transformers",
        "torch",
        "tqdm",
        "python-dotenv",
        "unstructured"
    ]
    
    logger.info("Проверка и установка необходимых зависимостей...")
    for package in required_packages:
        try:
            subprocess.run(
                ["pip", "install", package],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Установлен пакет: {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при установке пакета {package}: {e}")
            return False
    
    return True

def check_input_directory():
    """Проверка наличия директории с входными данными"""
    input_dir = Path("train_texts")
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Директория {input_dir} не существует или не является директорией")
        return False
    
    files = list(input_dir.glob("*.txt"))
    if not files:
        logger.error(f"В директории {input_dir} нет текстовых файлов")
        return False
    
    logger.info(f"Найдено {len(files)} текстовых файлов в директории {input_dir}")
    return True

def setup_google_api_key():
    """Проверка и настройка Google API ключа"""
    if "GOOGLE_API_KEY" not in os.environ:
        api_key = input("Введите ваш Google API ключ: ")
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Сохранение ключа в .env файл для последующего использования
        with open(".env", "a") as f:
            f.write(f"\nGOOGLE_API_KEY={api_key}\n")
        
        logger.info("Google API ключ сохранен в .env файл")
    
    return os.environ["GOOGLE_API_KEY"]

def run_synthetic_data_generation():
    """Запуск генерации синтетических данных"""
    logger.info("Запуск генерации синтетических данных...")
    try:
        subprocess.run(
            ["python", "generate_synthetic_data.py"],
            check=True
        )
        logger.info("Генерация синтетических данных завершена успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при генерации синтетических данных: {e}")
        return False

def run_rag_optimization():
    """Запуск оптимизации RAG"""
    logger.info("Запуск оптимизации RAG...")
    try:
        subprocess.run(
            ["python", "custom_rag_builder.py"],
            check=True
        )
        logger.info("Оптимизация RAG завершена успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при оптимизации RAG: {e}")
        return False

def main():
    logger.info("Запуск полного процесса настройки RAG...")
    
    # Проверка зависимостей
    if not check_dependencies():
        logger.error("Ошибка при установке зависимостей. Процесс остановлен.")
        return
    
    # Проверка входной директории
    if not check_input_directory():
        logger.error("Проблема с входной директорией. Процесс остановлен.")
        return
    
    # Настройка Google API ключа
    setup_google_api_key()
    
    # Генерация синтетических данных
    if not run_synthetic_data_generation():
        logger.error("Ошибка при генерации синтетических данных. Процесс остановлен.")
        return
    
    # Оптимизация RAG
    if not run_rag_optimization():
        logger.error("Ошибка при оптимизации RAG. Процесс остановлен.")
        return
    
    logger.info("Весь процесс успешно завершен!")
    logger.info("Оптимизированная конфигурация RAG сохранена в директории 'optimized_rag_config'")
    logger.info("Синтетические данные сохранены в файле 'synthetic_qa_data.csv'")
    logger.info("API-сервер запущен и доступен для использования")

if __name__ == "__main__":
    main() 