# Aurora_Assessment

# RAG QA System with FastAPI and Streamlit

This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI as the backend. The system fetches messages from an API, converts them into vector embeddings, and allows users to query the data to get AI-generated answers.

## Project Overview

- Initially, the approach was to use **TF-IDF vector search** to find relevant messages from the dataset. This worked for basic keyword matching but lacked the ability to generate natural, conversational answers.  
- To improve the user experience and create a more **question-and-answer feel**, we shifted to using an **open-source LLM** with embeddings stored in a **FAISS vector database**.  
- The system now combines **vector search** for context retrieval with an **LLM** to generate coherent, human-like answers based on the most relevant messages.

## Features

1. Fetch messages from an external API.
2. Convert messages into embeddings using `sentence-transformers`.
3. Store embeddings in a **FAISS** index for efficient similarity search.
4. Accept user queries and return **LLM-generated answers** using context from top similar vectors.

## Usage

1. Run the FastAPI backend:
    ```bash
    uvicorn main:app --reload --port 9000
    ```
2. Enter a query in the URL and get AI-generated answers.


<img width="972" height="162" alt="pic1" src="https://github.com/user-attachments/assets/97eb7abd-be8a-4a32-b37a-1c736aa1da16" />
<img width="895" height="181" alt="pic2" src="https://github.com/user-attachments/assets/6df8ca67-07ed-456c-a3a2-b4eab52b8d82" />
