from pydantic_settings import BaseSettings
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:

    class AppConfig(BaseSettings):
        directory_model_location: str = Field("./model", env="MODEL_DIR")

    def __init__(self):
        self.settings = self.AppConfig()
        self.model = AutoModelForCausalLM.from_pretrained(self.settings.directory_model_location, torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.directory_model_location)
