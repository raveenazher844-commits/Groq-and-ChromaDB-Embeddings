
# Groq + ChromaDB Embeddings Assignment

## Project Overview
This project demonstrates a simple Retrieval Augmented Generation (RAG)
workflow using Groq and ChromaDB.

## Features
- Loads text from a local knowledge file.
- Splits text into overlapping chunks.
- Generates text embeddings using ChromaDB's default embedding function.
- Stores and retrieves information from a persistent vector database.
- Uses the Groq API to generate answers from retrieved context.

## Technologies
- Python
- Groq API
- ChromaDB
- Python Dotenv

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Create a `.env` file and add your Groq API key.
4. Add information to `knowledge.txt`.
5. Run `python main.py`.

## Run
```bash
python main.py
```

## Security
Never commit your `.env` file or API key to GitHub.

## Author
Student Assignment Project
