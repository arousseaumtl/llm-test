import logging
from fastapi import FastAPI
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from typing import Optional
import torch
from dotenv import load_dotenv
from llm import LLM

load_dotenv()

class API:

    class AppConfig(BaseSettings):
        log_level: str = Field("INFO", env="LOG_LEVEL")

    class TextRequest(BaseModel):
        text: str
        max_length: int = 50
        clean_up_tokenization_spaces: Optional[bool] = True
        skip_special_tokens: Optional[bool] = True

    def __init__(self):
        self.settings = self.AppConfig()
        self.setup_logging()
        self.app = FastAPI()
        self.llm = LLM()
        self.setup_routes()

    def setup_logging(self):
        logging.basicConfig(level=self.settings.log_level.upper())
        self.logger = logging.getLogger(__name__)

    def setup_routes(self):

        @self.app.post("/generate")
        async def generate_response(request: self.TextRequest):
            inputs = self.llm.tokenizer.encode(request.text, return_tensors="pt")
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)

            outputs = self.llm.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=request.max_length,
                do_sample=True,
                top_p=0.95,
                top_k=60,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
            generated_text = self.llm.tokenizer.decode(
                outputs[0],
                skip_special_tokens=request.skip_special_tokens,
                clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
            )

            return {"generated_text": generated_text}

api = API()
app = api.app
