import gradio as gr
import os
from utils import clean_text, chunk_text, summarize_chunks
from config import get_api_key
import requests

# Получение API-ключа
TOGETHER_API_KEY = get_api_key()

# Промт для модели
PROMPT = (
    "Сгенерируй краткое саммари на русском (3-5 предложений) для следующего текста, "
    "сохраняя ключевые факты:\n\n{TEXT}"
)

# Функция для отправки запроса к Together.ai (Mistral-7B-Instruct)
def query_together_ai(prompt, api_key=TOGETHER_API_KEY):
    """
    Отправляет запрос к Together.ai (Mistral-7B-Instruct) и возвращает ответ модели.
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

# Основная функция саммаризации
def summarize_interface(text, file):
    # Получение текста из поля или файла
    if file is not None:
        text = file.read().decode('utf-8')
    if not text or text.strip() == '':
        return 'Пожалуйста, введите текст или загрузите файл.'

    # 1. Очистка текста
    cleaned = clean_text(text)
    # 2. Разбиение на чанки
    chunks = chunk_text(cleaned)
    # 3. Многоэтапная генерация саммари
    summary = summarize_chunks(chunks, query_together_ai, PROMPT)
    return summary

# Gradio-интерфейс
demo = gr.Interface(
    fn=summarize_interface,
    inputs=[
        gr.Textbox(label="Введите текст для саммари", lines=10),
        gr.File(label="или загрузите .txt файл", file_types=[".txt"])
    ],
    outputs=gr.Textbox(label="Саммари"),
    title="🎮 Текстовый саммаризатор для Ifortex ML Intern (2025)",
    description="Генерация краткого саммари длинных текстов с помощью Mistral-7B-Instruct (Together.ai)",
)

def main():
    demo.launch()

if __name__ == "__main__":
    main() 