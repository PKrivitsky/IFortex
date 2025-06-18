import re
import tiktoken

# Очистка текста
def clean_text(text):
    """
    Удаляет лишние пробелы, спецсимволы и приводит текст к нормальному виду.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Подсчет количества токенов в тексте
def count_tokens(text, model_name='gpt2'):
    enc = tiktoken.get_encoding(model_name)
    return len(enc.encode(text))

def chunk_text(text, max_tokens=2000, overlap=200, model_name='gpt2'):
    """
    Разбивает текст на чанки с учетом границ предложений и ограничений по токенам.
    """
    if not text or not text.strip():
        return []
        
    # Инициализация токенизатора
    enc = tiktoken.get_encoding(model_name)
    
    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        # Токенизируем предложение
        sentence_tokens = enc.encode(sentence)
        sentence_token_count = len(sentence_tokens)
        
        # Если предложение слишком длинное, разбиваем его
        if sentence_token_count > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Разбиваем длинное предложение на части
            start = 0
            while start < sentence_token_count:
                end = min(start + max_tokens, sentence_token_count)
                chunk = enc.decode(sentence_tokens[start:end])
                chunks.append(chunk)
                start = end - overlap
        
        # Если добавление предложения превысит лимит, создаем новый чанк
        elif current_tokens + sentence_token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_token_count
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_token_count
    
    # Добавляем последний чанк, если он есть
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Проверяем количество чанков
    if len(chunks) > 10:
        combined_chunks = []
        current_combined = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(enc.encode(chunk))
            if current_tokens + chunk_tokens > max_tokens:
                combined_chunks.append(' '.join(current_combined))
                current_combined = [chunk]
                current_tokens = chunk_tokens
            else:
                current_combined.append(chunk)
                current_tokens += chunk_tokens
        
        if current_combined:
            combined_chunks.append(' '.join(current_combined))
        
        chunks = combined_chunks
    
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