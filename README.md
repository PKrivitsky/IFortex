# 🎮 Текстовый саммаризатор

Генерация краткого саммари длинных текстов с помощью Mistral-7B-Instruct (Together.ai) и Gradio.

## Установка

```bash
pip install -r requirements.txt
```

## Настройка
1. Получите Together.ai API-ключ: https://platform.together.ai/
2. Создайте файл `.env` и добавьте строку:
   ```
   TOGETHER_API_KEY=ваш_ключ
   ```

## Запуск

```bash
python main.py
```

## Использование
- Введите текст или загрузите .txt файл
- Нажмите "Сгенерировать"
- Получите краткое саммари

## Файлы
- `main.py` — основной код
- `utils.py` — обработка текста и чанков
- `.env` — ваш API-ключ (НЕ коммитьте!)
- `requirements.txt` — зависимости

---
