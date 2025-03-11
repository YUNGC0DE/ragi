#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для генерации синтетических вопросов и ответов на основе текстов
из директории train_texts с использованием Google модели.
"""

import os
import csv
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def load_text_files(directory: Path) -> List[Dict[str, str]]:
    """Загрузка текстовых файлов из директории"""
    documents = []
    for file_path in directory.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "source": file_path.name
                })
            logger.info(f"Загружен файл: {file_path.name}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {file_path}: {e}")
    
    logger.info(f"Всего загружено документов: {len(documents)}")
    return documents

def split_documents(documents: List[Dict[str, str]], 
                   chunk_size: int = 1500, 
                   chunk_overlap: int = 150) -> List[Dict[str, str]]:
    """Разбиение документов на чанки"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc["content"])
        for chunk in doc_chunks:
            chunks.append({
                "content": chunk,
                "source": doc["source"]
            })
    
    logger.info(f"Документы разбиты на {len(chunks)} чанков")
    return chunks

def generate_qa_pairs(chunks: List[Dict[str, str]], 
                     num_pairs: int = 50,
                     model_name: str = "gemini-1.5-pro") -> List[Dict[str, str]]:
    """Генерация пар вопрос-ответ на основе чанков"""
    # Инициализация модели
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Создание промпта для генерации вопросов и ответов
    qa_prompt = ChatPromptTemplate.from_template("""
    Ты эксперт по созданию вопросов и ответов для обучения RAG-системы.
    
    Вот фрагмент текста:
    
    ```
    {chunk_content}
    ```
    
    Создай 1 сложный и специфичный вопрос по этому тексту и подробный ответ на него.
    Вопрос должен требовать глубокого понимания текста и быть таким, чтобы ответ можно было найти только в этом фрагменте.
    
    Формат ответа:
    Вопрос: [вопрос]
    Ответ: [ответ]
    """)
    
    # Создание цепочки для генерации
    qa_chain = qa_prompt | llm | StrOutputParser()
    
    # Выбор случайных чанков
    selected_chunks = random.sample(chunks, min(num_pairs, len(chunks)))
    
    # Генерация пар вопрос-ответ
    qa_pairs = []
    for chunk in tqdm(selected_chunks, desc="Генерация вопросов и ответов"):
        try:
            result = qa_chain.invoke({"chunk_content": chunk["content"]})
            
            # Парсинг результата
            lines = result.strip().split('\n')
            question = ""
            answer = ""
            
            for line in lines:
                if line.startswith("Вопрос:"):
                    question = line[len("Вопрос:"):].strip()
                elif line.startswith("Ответ:"):
                    answer = line[len("Ответ:"):].strip()
            
            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "source": chunk["source"],
                    "context": chunk["content"]
                })
        except Exception as e:
            logger.error(f"Ошибка при генерации пары вопрос-ответ: {e}")
    
    logger.info(f"Сгенерировано {len(qa_pairs)} пар вопрос-ответ")
    return qa_pairs

def save_to_csv(qa_pairs: List[Dict[str, str]], output_file: str = "synthetic_qa_data.csv"):
    """Сохранение пар вопрос-ответ в CSV файл"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "source", "context"])
        writer.writeheader()
        writer.writerows(qa_pairs)
    
    logger.info(f"Данные сохранены в файл: {output_file}")

def main():
    # Настройка Google API ключа
    setup_google_api_key()
    
    # Путь к директории с текстами
    input_dir = Path("train_texts")
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Директория {input_dir} не существует или не является директорией")
    
    # Загрузка текстовых файлов
    documents = load_text_files(input_dir)
    
    # Разбиение документов на чанки
    chunks = split_documents(documents)
    
    # Генерация пар вопрос-ответ
    qa_pairs = generate_qa_pairs(chunks, num_pairs=50)
    
    # Сохранение результатов
    save_to_csv(qa_pairs)

if __name__ == "__main__":
    main() 