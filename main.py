from models.embedder import Embedder
from models.llm import RAG_LLM
from vector_store.weaviate_store import Weaviate_Store

from utils.pdf_splitter import PDFSplitter
from utils.build_graph import Knowledge_graph

import torch

if __name__ == '__main__':
    splitter = PDFSplitter()
    chunks = splitter.split_document(r"C:\Users\Jino Rohit\Downloads\Jino-Rohit-Resume.pdf", max_size = 500)

    rag_llm = RAG_LLM(
        model_directory="C:/Users/Jino Rohit/Downloads/mistral-7b-orca",
        temperature=1.0,
        top_k=5,
        top_p=0.8,
        top_a=0.9,
        token_repetition_penalty=1.2
    )
    rag_llm.setup_model()
    entities_df = rag_llm.generate_nodes(chunks, max_new_tokens = 1000)

    kg = Knowledge_graph(entities_df)
    kg.create_graph()

    #if you want to do hybrid search, create embeddings
    emb_model = Embedder()
    nodes_embeds = emb_model.embed(list(entities_df['node_1'] + ' ' + entities_df['node_2'] + ' ' + entities_df['edge']))

    entities_df['vectors'] = nodes_embeds.tolist()

    vector_store = Weaviate_Store(store_name = 'Demo')
    vector_store.store_vectors(entities_df)

    query = "what was the solution for the hackathon"
    response = vector_store.keyword_search(query = query, top_k = 5)

    for _r in response['data']['Get']['Demo']:
        kg.query_sub_graph(_r['source'])
    
    rag_llm.generate_answers(chunks, query, 500)

    
