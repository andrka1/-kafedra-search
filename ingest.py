"""
Индексация документов из папки docs/ в векторную базу ChromaDB.

Поддерживаемые форматы: PDF, DOCX, TXT, MD.
Запуск: python ingest.py
"""

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = str(BASE_DIR / "vecdb")
COLLECTION_NAME = "kafedra"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    if ext in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    return []


def main() -> None:
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"⚠️  Папка {DOCS_DIR} была создана — положи туда файлы и запусти снова.")
        return

    files = [
        p for p in DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
    ]
    print(f"📂 Найдено файлов в папке docs: {len(files)}")
    if not files:
        print("⚠️  Нет файлов для индексации. Положи PDF/DOCX/TXT/MD в папку docs/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_texts: list[str] = []
    all_meta: list[dict] = []

    for file_path in files:
        try:
            docs = load_file(file_path)
        except Exception as e:
            print(f"  ❌ Пропускаю {file_path.name}: {e}")
            continue
        if not docs:
            print(f"  ⚠️  Пропускаю {file_path.name} (пусто)")
            continue

        chunks = splitter.split_documents(docs)
        for c in chunks:
            all_texts.append(c.page_content)
            meta = {"source": file_path.name}
            if "page" in c.metadata:
                meta["page"] = c.metadata["page"]
            all_meta.append(meta)
        print(f"  ✅ {file_path.name} — {len(docs)} страниц(ы), {len(chunks)} чанков")

    if not all_texts:
        print("⚠️  Нечего индексировать.")
        return

    print(f"\n🧠 Загружаю модель эмбеддингов: {EMBEDDING_MODEL}")
    print("   (первый раз качается ~500 МБ, потом кэшируется)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"🔢 Считаю эмбеддинги для {len(all_texts)} фрагментов...")
    # Префикс 'passage:' — обязательно для e5-моделей
    embeddings = model.encode(
        [f"passage: {t}" for t in all_texts],
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32,
    ).tolist()

    print("💾 Сохраняю в ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    # Пересоздаём коллекцию, чтобы избежать дубликатов
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(all_texts))]
    collection.add(
        ids=ids,
        documents=all_texts,
        metadatas=all_meta,
        embeddings=embeddings,
    )

    print(f"\n✅ Готово! Проиндексировано {len(all_texts)} чанков из {len(files)} файлов.")
    print(f"   База: {DB_DIR}")
    print("\n▶️  Следующий шаг: uvicorn web.app:app --reload")


if __name__ == "__main__":
    main()
