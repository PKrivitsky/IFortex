import re
import tiktoken

# Очистка текста от лишних символов
def clean_text(text):
    """
    Удаляет лишние пробелы, спецсимволы и приводит текст к нормальному виду.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Подсчет количества токенов в тексте (Mistral = gpt2)
def count_tokens(text, model_name='gpt2'):
    enc = tiktoken.get_encoding(model_name)
    return len(enc.encode(text))

# Разбиение текста на чанки по ~4000 токенов с overlap
def chunk_text(text, max_tokens=4000, overlap=200, model_name='gpt2'):
    enc = tiktoken.get_encoding(model_name)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

# Итеративная генерация саммари для списка чанков
def summarize_chunks(chunks, query_func, prompt_template):
    """
    Для каждого чанка вызывает query_func, затем объединяет саммари и повторяет до финального результата.
    """
    summaries = []
    for chunk in chunks:
        prompt = prompt_template.replace('{TEXT}', chunk)
        summary = query_func(prompt)
        summaries.append(summary)
    # Если чанков > 1, сжимаем итоговое саммари
    while len(summaries) > 1:
        joined = '\n'.join(summaries)
        new_chunks = chunk_text(joined)
        summaries = [query_func(prompt_template.replace('{TEXT}', c)) for c in new_chunks]
    return summaries[0] if summaries else '' 