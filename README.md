# 🎓 ИИ-помощник кафедры ИТ

Прототип семантического поиска и ответов на вопросы студентов по учебным материалам кафедры.
Стек: **Python + FastAPI + ChromaDB + Tailwind CSS**.

Два режима:
- 🔎 **Оригинальный поиск** — карточки с фрагментами и процентом релевантности. Бесплатно.
- ⚡ **Умный ответ (ИИ)** — связный ответ через GPT-4o-mini + источники. Требует API-ключ OpenAI.

## 🚀 Запуск (Windows 10, PowerShell)

```powershell
# 1. Создать виртуальное окружение
python -m venv venv

# 2. Активировать (если ругается — Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
venv\Scripts\activate

# 3. Установить библиотеки (первый раз — 3–5 минут)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Проиндексировать документы из папки docs/
python ingest.py

# 5. Запустить веб-сервер
uvicorn web.app:app --reload
```

Открой в браузере: **http://localhost:8000**

Остановить сервер: `Ctrl + C` в терминале.

## 🤖 Подключение OpenAI (опционально)

1. Получи ключ: https://platform.openai.com/api-keys
2. Переименуй `.env.example` в `.env`
3. Впиши туда: `OPENAI_API_KEY=sk-...`
4. Перезапусти сервер

Стоимость: ~$0.001 за вопрос (gpt-4o-mini).

## 🧩 Структура

```
kafedra-search/
├── docs/                     ← учебные материалы (PDF, DOCX, TXT, MD)
├── web/
│   ├── app.py                ← FastAPI-сервер
│   └── static/index.html     ← веб-интерфейс
├── ingest.py                 ← индексация
├── search.py                 ← CLI-поиск (для отладки)
├── requirements.txt
├── .env.example
└── README.md
```

## 🔗 API

| Метод | Адрес | Описание |
|---|---|---|
| GET  | `/`          | Веб-интерфейс |
| POST | `/api/ask`   | `{ question, mode: "raw"\|"ai" }` |
| GET  | `/api/stats` | Статистика базы |
| GET  | `/docs`      | Swagger UI |
