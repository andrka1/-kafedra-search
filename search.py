"""
Поиск в терминале (CLI-версия, для отладки без веб-сервера).

Запуск: python search.py
Выйти: Enter на пустой строке или Ctrl+C

Для красивого интерфейса запусти: uvicorn web.app:app --reload
"""

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = str(BASE_DIR / "vecdb")
COLLECTION_NAME = "kafedra"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 5


def answer_with_openai(question: str, contexts: list[str]) -> str:
    from openai import OpenAI

    client = OpenAI()
    context = "\n\n---\n\n".join(contexts)
    prompt = f"""Ты — ассистент кафедры, помогаешь студентам разобраться в учебном материале.
Ответь на вопрос студента, опираясь ТОЛЬКО на фрагменты ниже.
Если в материалах нет ответа — так и скажи.

=== Фрагменты ===
{context}

=== Вопрос ===
{question}

=== Ответ ==="""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    if not Path(DB_DIR).exists():
        print("⚠️  Сначала запусти: python ingest.py")
        return

    print("🧠 Загружаю модель эмбеддингов...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    use_ai = bool(os.getenv("OPENAI_API_KEY"))
    print("\n🎓 ИИ-помощник кафедры готов к работе")
    print(f"   Фрагментов в базе: {collection.count()}")
    print(f"   Режим: {'Умные ответы (OpenAI)' if use_ai else 'Только фрагменты (без OpenAI)'}")
    print("   Пустая строка или Ctrl+C — выход\n")

    while True:
        try:
            question = input("❓ Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Пока!")
            break
        if not question:
            print("👋 Пока!")
            break

        print("\n🔎 Ищу в материалах...")
        q_emb = model.encode(
            f"query: {question}", normalize_embeddings=True
        ).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=TOP_K)

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if use_ai:
            print("🤖 Формирую ответ...\n")
            answer = answer_with_openai(question, docs)
            print("📝 Ответ:\n")
            print(answer)
            print("\n📚 Источники:")
            for m in metas:
                print(f"  • {m.get('source', '?')}")
        else:
            print("📚 Найденные фрагменты:\n")
            for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                print(f"--- Фрагмент {i} (из «{meta.get('source', '?')}») ---")
                print(doc)
                print()

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
