# RAG with Knowledge Graph

## Overview
Retrieval Augmented Generation(RAG) is a way of generating reliable answers from LLM using an external knowledge base.
This project shows how to use RAG with a knowledge graph using Weaviate as the vector database and the exllamav2 implementation of the mistral orca model.

The following is the pipeline -
1. Extract text from a PDF
2. Chunk the data into k size with w overlap.
2. Extract (source, relation, target) from the chunks and create a knowledge graph
3. Extract embeddings for the nodes and relationships.
4. Store the text and vectors in weaviate vector database.
5. Apply a keyword search on the nodes and retrieve the top k chunks.
6. Generate the answer from the top k retrieved chunks.
7. You can also visualize the sub-graph of the nodes used to generate the answer.

## Installation

```bash
pip install -r requirements.txt
```

## Features
- Uses the Exllamav2 implementation of the mistral orca model which is extremely fast and memory efficient.
- Construction of knowledge graph from text to understand the concepts better and retrieve more relevant text chunks.
- The vector database used is Weaviate for storing the data and applying the keyword search(can also be done for hybrid search).
- 

## Usage

- First signup to weaviate free sandbox to get your api key and weaviate instance url.
- Create a .env file and store your credentials there.

Sample .env 

```env
WEAVIATE_API_KEY = 'xxx'
WEAVIATE_CLIENT_URL = 'xxx'
```

- Run the main file.

```bash
python main.py
```
