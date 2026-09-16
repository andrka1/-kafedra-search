"""
FastAPI-сервер для ИИ-помощника кафедры.

Эндпоинты:
- GET  /             — отдаёт веб-интерфейс (index.html)
- POST /api/ask      — принимает вопрос + mode, возвращает фрагменты или ИИ-ответ
- GET  /api/stats    — статистика базы
- GET  /api/health   — проверка готовности

Запуск: uvicorn web.app:app --reload
Открыть в браузере: http://localhost:8000
"""

import os
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = str(BASE_DIR / "vecdb")
STATIC_DIR = Path(__file__).resolve().parent / "static"
COLLECTION_NAME = "kafedra"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 5

app = FastAPI(
    title="ИИ-помощник кафедры",
    description="Семантический поиск по учебным материалам",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Optional[SentenceTransformer] = None
_collection = None


@app.on_event("startup")
def startup() -> None:
    global _model, _collection
    if not Path(DB_DIR).exists():
        print(
            f"\n⚠️  Векторная база не найдена в {DB_DIR}.\n"
            f"   Запусти сначала:  python ingest.py\n"
        )
        return

    print("🧠 Загружаю модель эмбеддингов...")
    _model = SentenceTransformer(EMBEDDING_MODEL)

    print("📂 Подключаюсь к векторной базе...")
    client = chromadb.PersistentClient(path=DB_DIR)
    _collection = client.get_collection(COLLECTION_NAME)

    print(f"✅ Готово! Фрагментов в базе: {_collection.count()}")
    print("🌐 Открой в браузере: http://localhost:8000\n")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["raw", "ai"] = "raw"


class Fragment(BaseModel):
    text: str
    source: str
    page: Optional[int] = None
    score: float


class AskResponse(BaseModel):
    mode: str
    question: str
    fragments: list[Fragment]
    answer: Optional[str] = None
    openai_used: bool = False


def _search(question: str) -> list[Fragment]:
    if _model is None or _collection is None:
        raise HTTPException(
            status_code=503,
            detail="Сервер не готов: векторная база не загружена. Запусти python ingest.py.",
        )

    query_embedding = _model.encode(
        f"query: {question}", normalize_embeddings=True
    ).tolist()

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    fragments: list[Fragment] = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))

        raw_page = meta.get("page")
        page: Optional[int] = None
        if isinstance(raw_page, int):
            page = raw_page + 1
        elif isinstance(raw_page, str) and raw_page.isdigit():
            page = int(raw_page) + 1

        fragments.append(
            Fragment(
                text=text,
                source=str(meta.get("source", "неизвестно")),
                page=page,
                score=round(score, 3),
            )
        )
    return fragments


def _build_prompt(question: str, fragments: list[Fragment]) -> str:
    context = "\n\n---\n\n".join(
        f"[Источник: {f.source}]\n{f.text}" for f in fragments
    )
    return f"""Ты — ассистент кафедры, помогаешь студентам разобраться в учебном материале.
Ответь на вопрос студента, опираясь ТОЛЬКО на приведённые ниже фрагменты из учебных материалов.
Если в материалах нет ответа — честно скажи: «В материалах кафедры я не нашёл ответа на этот вопрос».
Пиши ясно и структурно. Источники не перечисляй — они показываются отдельно в интерфейсе.

=== Фрагменты из учебных материалов ===
{context}

=== Вопрос студента ===
{question}

=== Ответ ==="""


def _answer_with_openai(question: str, fragments: list[Fragment]) -> str:
    from openai import OpenAI

    client = OpenAI()
    prompt = _build_prompt(question, fragments)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    fragments = _search(request.question)

    if request.mode == "ai":
        if not os.getenv("OPENAI_API_KEY"):
            return AskResponse(
                mode="ai",
                question=request.question,
                fragments=fragments,
                answer=(
                    "⚠️ Режим «Умный ответ» требует ключ OpenAI. "
                    "Добавь OPENAI_API_KEY в файл .env и перезапусти сервер. "
                    "Пока вот найденные фрагменты из материалов кафедры."
                ),
                openai_used=False,
            )
        try:
            answer = _answer_with_openai(request.question, fragments)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")
        return AskResponse(
            mode="ai",
            question=request.question,
            fragments=fragments,
            answer=answer,
            openai_used=True,
        )

    return AskResponse(
        mode="raw",
        question=request.question,
        fragments=fragments,
        openai_used=False,
    )


@app.get("/api/stats")
def stats() -> dict:
    if _collection is None:
        return {"ready": False, "chunks": 0, "documents": 0}
    docs_dir = BASE_DIR / "docs"
    doc_count = 0
    if docs_dir.exists():
        doc_count = sum(
            1
            for p in docs_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
        )
    return {
        "ready": True,
        "chunks": _collection.count(),
        "documents": doc_count,
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "ready": _collection is not None}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html не найден")
    return FileResponse(str(index_path))
