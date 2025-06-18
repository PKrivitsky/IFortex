import gradio as gr
import os
from utils import clean_text, chunk_text, summarize_chunks, count_tokens
from config import get_api_key
import requests
import docx
import PyPDF2
import io

# Получение API-ключа
TOGETHER_API_KEY = get_api_key()

# Промт для модели
PROMPT = """Ты - эксперт по созданию качественных саммари текстов. Твоя задача - создать краткое, но информативное саммари, которое:

1. Сохраняет все ключевые факты и идеи
2. Поддерживает логическую структуру оригинала
3. Использует четкий и профессиональный язык
4. Исключает повторения и несущественные детали
5. Сохраняет важные термины и определения
6. Отражает основной тон и стиль оригинала

Создай саммари длиной 3-5 предложений, которое будет понятно даже читателю, не знакомому с темой.

Текст для саммаризации:
{TEXT}

Саммари:"""


def query_together_ai(prompt, api_key=TOGETHER_API_KEY):
    """
    Функция для отправки запроса к Together.ai (Mistral-7B-Instruct) и получения ответа модели.
    """
    url = "https://api.together.xyz/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['text'].strip()
    except Exception as e:
        return f"[Ошибка запроса к Together.ai: {e}]"

def get_text_from_file(file):
    """Извлекает текст из файла разных форматов"""
    if file is None:
        return None
        
    file_name = file.name.lower()
    
    try:
        if file_name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_name.endswith('.docx'):
            doc = docx.Document(file.name)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        elif file_name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file.name)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text
        else:
            return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {str(e)}")
        return None

def get_text_statistics(text):
    """Возвращает статистику по тексту"""
    words = text.split()
    sentences = text.split('.')
    tokens = count_tokens(text)
    return f"Слов: {len(words)}, Предложений: {len(sentences)}, Токенов: {tokens}"

# Основная функция саммаризации
def summarize_interface(text, file):
    # Получение текста 
    if file is not None:
        file_text = get_text_from_file(file)
        if file_text:
            text = file_text
        else:
            return 'Ошибка при чтении файла. Пожалуйста, проверьте формат файла.', '', ''
            
    if not text or text.strip() == '':
        return 'Пожалуйста, введите текст или загрузите файл.', '', ''

    # Получаем статистику
    stats = get_text_statistics(text)
    
    # 1. Очистка текста
    cleaned = clean_text(text)
    # 2. Разбиение на чанки
    chunks = chunk_text(cleaned)
    # 3. Генерация саммари
    summary = summarize_chunks(chunks, query_together_ai, PROMPT)
    
    return text, summary, stats

# Gradio-интерфейс
demo = gr.Interface(
    fn=summarize_interface,
    inputs=[
        gr.Textbox(
            label="Введите текст для саммари", 
            lines=10,
            placeholder="Вставьте сюда текст статьи, отчета или другого документа..."
        ),
        gr.File(
            label="или загрузите файл", 
            file_types=[".txt", ".doc", ".docx", ".pdf"],
            file_count="single"
        )
    ],
    outputs=[
        gr.Textbox(label="Исходный текст", lines=10),
        gr.Textbox(label="Саммари", lines=5),
        gr.Textbox(label="Статистика", lines=2)
    ],
    title="Текстовый саммаризатор",
    description="""
    Генерация краткого саммари для различных текстовых документов:
    - Новостные статьи
    - Отчёты и посты
    - Материалы с Википедии
    - Загруженные файлы (.txt, .doc, .docx, .pdf)
    
    Поддерживает как короткие (~500 слов), так и длинные тексты (до 5000+ слов).
    """,
    examples=[
        ["Это пример короткого текста для саммаризации. Здесь может быть любой текст, который вы хотите сократить до нескольких предложений.", None]
    ],
    submit_btn=gr.Button("Сгенерировать саммари", variant="primary"),
    clear_btn=gr.Button("Очистить", variant="secondary")
)

def main():
    demo.launch()

if __name__ == "__main__":
    main() 