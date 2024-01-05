from pydantic import BaseModel

class InputData(BaseModel):
    model_directory: str = 'None'
    temperature: float
    top_k: float
    top_p: float
    top_a: float
    token_repetition_penalty: float