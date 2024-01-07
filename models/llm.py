import sys, os
from loguru import logger
from utils.pdf_splitter import PDFSplitter

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import re
import pandas as pd

def lowercase_dict(d):
    return {key: value.lower() for key, value in d.items()}


def process_results(object):
    filtered_list = [item for item in object if len(item) > 1]

    elements_to_remove = {
        'node_1': 'A concept from extracted ontology',
        'node_2': 'A related concept from extracted ontology',
        'edge': 'relationship between the two concepts, node_1 and node_2 in one or two sentences'
    }

    filtered_list = [lowercase_dict(item) for item in filtered_list if item != elements_to_remove]
    return filtered_list

system_prompt = """
You are a network graph maker who extracts terms and their relations from a given context. 
You are provided with a context chunk (delimited by ```) Your task is to extract the ontology
of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n
Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n
\tTerms may include object, entity, location, organization, person, \n
\tcondition, acronym, documents, service, concept, etc.\n
\tTerms should be as atomistic as possible\n\n
Thought 2: Think about how these terms can have one on one relation with other terms.\n
\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n
\tTerms can be related to many other terms\n\n
Thought 3: Find out the relation between each such related pair of terms. \n\n
Format your output as a list of json. Each element of the list contains a pair of terms
and the relation between them, like the following: \n
[\n
    {\n
        "node_1": "A concept from extracted ontology",\n
        "node_2": "A related concept from extracted ontology",\n
        "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n
    }, {...}\n"
]"
DO NOT RETURN ANY EXPLANATION, ONLY RETURN THE LIST OF JSON.
"""

qna_prompt = """You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'.
You only respond once as Assistant. You are allowed to use only the given context below to answer the user's queries, 
and if the answer is not present in the context, say you don't know the answer.
CONTEXT: {context}
"""

class RAG_LLM:
    def __init__(self, model_directory: str, temperature: float, top_k: float, top_p: float, top_a: float, token_repetition_penalty: float):

        self.model_directory = model_directory
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.top_a = top_a
        self.token_repetition_penalty = token_repetition_penalty

        #r"C:\Users\Jino Rohit\Downloads\mistral-7b-orca"
    
    def setup_model(self) -> None:
        self.config = ExLlamaV2Config()
        self.config.model_dir = self.model_directory
        self.config.prepare()

        self.model = ExLlamaV2(self.config)
        logger.info("Loading model...")

        self.cache = ExLlamaV2Cache(self.model, lazy = True)
        self.model.load_autosplit(self.cache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)

        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = self.temperature
        self.settings.top_k = self.top_k
        self.settings.top_p = self.top_p
        self.settings.top_a = self.top_a
        self.settings.token_repetition_penalty = self.token_repetition_penalty
        self.settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

    def generate_nodes(self, chunks, max_new_tokens) -> str:
        if self.generator is None or self.settings is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        all_matches = []

        self.generator.warmup()

        prompt = """<|im_start|>system
        {system_prompt}
        <|im_end|>
        <|im_start|>user
        {text_chunk}
        <|im_end|>
        <|im_start|>assistant
        """

        for idx, chunk in enumerate(chunks):
            logger.info(f"Extracting tuples from chunk: {idx}")
            output = self.generator.generate_simple(prompt.format(system_prompt = system_prompt, text_chunk = chunk['text']), self.settings, max_new_tokens, seed = 1234)
            
            #extracting dict types
            pattern = r'\{[^}]+\}'
            matches = re.findall(pattern, output)
            try:
                dictionaries = [eval(match) for match in matches]
                dictionaries = process_results(dictionaries)

                for _d in dictionaries:
                    _d['chunk'] = chunk['text']
                    
                all_matches.extend(dictionaries)
            except:
                pass
        
        df = pd.DataFrame(all_matches)
        df = df.drop_duplicates(subset=['node_1', 'node_2', 'edge'], keep=False)
        return df
    
    def generate_answers(self, chunks, query, max_new_tokens) -> str:
        if self.generator is None or self.settings is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        self.generator.warmup()

        prompt = """<|im_start|>system
        {qna_prompt}
        <|im_end|>
        <|im_start|>user
        {query}
        <|im_end|>
        <|im_start|>assistant
        """
        logger.info(f"Asking the assistant : {query}")
        output = self.generator.generate_simple(prompt.format(qna_prompt = qna_prompt.format(context = chunks), query = query), self.settings, max_new_tokens, seed = 1234)

        start_tag = "<|im_start|>"
        end_tag = "<|im_end|>"
        start_index = output.rfind(start_tag)
        end_index = output.rfind(end_tag)
        logger.info(f"Answer : {output[start_index + len(start_tag): end_index]}")

        



# if __name__ == '__main__':
#     rag_llm_instance = RAG_LLM(
#         model_directory="C:/Users/Jino Rohit/Downloads/mistral-7b-orca",
#         temperature=1.0,
#         top_k=5,
#         top_p=0.8,
#         top_a=0.9,
#         token_repetition_penalty=1.2
#     )

#     rag_llm_instance.setup_model()

#     prompt = "write a story on"
#     generated_text = rag_llm_instance.generate_text(prompt, 100)
#     print(generated_text)