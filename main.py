
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq

# Load API key and settings from the .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"
DATABASE_DIR = BASE_DIR / "chroma_db"

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Check that the API key exists
if not API_KEY or API_KEY == "paste_your_actual_api_key_here":
    raise ValueError(
        "Please add your real GROQ_API_KEY to the .env file."
    )

# Create the Groq client
groq_client = Groq(api_key=API_KEY)

# Connect to a persistent ChromaDB database
chroma_client = chromadb.PersistentClient(
    path=str(DATABASE_DIR)
)

# Create or open the knowledge collection
collection = chroma_client.get_or_create_collection(
    name="assignment_knowledge"
)


def load_knowledge():
    """Read the knowledge file and split it into chunks."""

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            "knowledge.txt was not found in the project folder."
        )

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("knowledge.txt is empty.")

    # Split text into overlapping chunks
    chunk_size = 500
    overlap = 80
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def store_embeddings():
    """Generate embeddings and store text in ChromaDB."""

    chunks = load_knowledge()

    # Remove old records so the latest file is indexed
    old_records = collection.get()

    if old_records["ids"]:
        collection.delete(ids=old_records["ids"])

    # ChromaDB generates embeddings automatically
    # when documents are added to the collection.
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
    )

    print(f"\nKnowledge chunks stored: {len(chunks)}")
    print(f"Database location: {DATABASE_DIR}")


def retrieve_context(question):
    """Search ChromaDB for relevant knowledge."""

    results = collection.query(
        query_texts=[question],
        n_results=min(3, collection.count()),
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "No relevant information was found."

    return "\n\n".join(documents)


def ask_question(question):
    """Retrieve context and ask Groq to generate an answer."""

    context = retrieve_context(question)

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assignment assistant. "
                    "Answer using the supplied knowledge context. "
                    "If the context does not contain the answer, "
                    "say that the information is not available "
                    "in the knowledge base. Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Knowledge context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=500,
    )

    answer = response.choices[0].message.content
    return answer or "No answer was returned."


def main():
    print("=" * 50)
    print(" GROQ + CHROMADB EMBEDDINGS ASSIGNMENT")
    print("=" * 50)

    store_embeddings()

    print("\nAsk a question about the knowledge file.")
    print("Type 'exit' to close the application.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            print("\nSearching knowledge and generating answer...")
            answer = ask_question(question)
            print(f"\nAssistant: {answer}\n")
        except Exception as error:
            print(f"\nAn error occurred: {error}\n")


if __name__ == "__main__":
    main()
