import sys, os

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
        print("Loading model: " + self.model_directory)

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
    
    def generate_text(self, prompt, max_new_tokens) -> str:
        if self.generator is None or self.settings is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        output = self.generator.generate_simple(prompt, self.settings, max_new_tokens, seed = 1234)
        return output


if __name__ == '__main__':
    rag_llm_instance = RAG_LLM(
        model_directory="C:/Users/Jino Rohit/Downloads/mistral-7b-orca",
        temperature=1.0,
        top_k=5,
        top_p=0.8,
        top_a=0.9,
        token_repetition_penalty=1.2
    )

    rag_llm_instance.setup_model()

    prompt = "write a story on"
    generated_text = rag_llm_instance.generate_text(prompt, 100)
    print(generated_text)