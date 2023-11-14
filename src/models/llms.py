from enum import Enum
from typing import Any, Optional

import together
from backoff import on_exception, expo
from config import Config
from langchain.llms import Together
from langchain.llms.together import Together
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel


class VendorModels(Enum):
    pass

class OpenAIModels(VendorModels):
    """
    Models availible from OpenAI
    Sadly, this is not availible in their library
    """

    gpt3_5_turbo_0613 = "gpt-3.5-turbo-0613"
    gpt3_5_turbo = "gpt-3.5-turbo"
    gpt3_5_turbo_16k = "gpt-3.5-turbo-16k"
    gpt3_5_ft = "ft:gpt-3.5-turbo-0613:personal::8G9xDV6J"
    gpt4 = "gpt-4"

class TogetherModels(VendorModels):
    llama2_7b = "togethercomputer/llama-2-7b"
    llama2_7b_chat = "togethercomputer/llama-2-7b-chat"

# class LLMResponse(BaseModel):
#     input: str
#     output: str
#     raw: dict[str, Any]

# class TogetherAI:
#     def __init__(self):
#         together.api_key = Config.TOGETHER_API_KEY
#         # self.model = "togethercomputer/RedPajama-INCITE-7B-Instruct"
#         self.model = TogetherModels.llama2_7b_chat.value


#     @on_exception(expo, RateLimitException, max_tries=8)
#     @limits(calls=1, period=5)
#     def chat(self, content: str) -> LLMResponse:
#         # print("together prompt")
#         # print(content + "\n\n")
#         output = together.Complete.create(
#             # prompt = "<human>: What are Isaac Asimov's Three Laws of Robotics?\n<bot>:",
#             prompt = content,
#             # model = "togethercomputer/RedPajama-INCITE-7B-Instruct",
#             model = self.model,
#             max_tokens = 256,
#             temperature = 0.8,
#             top_k = 60,
#             top_p = 0.6,
#             repetition_penalty = 1.1,
#             stop = ["[INST]", "None", "User:"]
#             # stop = ['<human>', '\n\n']
#         )

#         # return output['prompt'][0]+output['output']['choices'][0]['text']

#         response =  LLMResponse(
#             input=output['prompt'][0],
#             output=output['output']['choices'][0]['text'],
#             raw=output
#         )

#         return response

#     def to_json(self):
#         return self.__dict__

class LLMFactory:
    """
    Methods to make it easy to switch between different LLMs defined in Langchain.
    Determines the LLM's type based on model enum.
    """
    @classmethod
    def chat(cls,
        model: VendorModels,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None
    ) -> BaseChatModel:
        _kwargs = {
            "max_tokens": max_tokens,
            "model": model.value,
            "n": n,
            "temperature": temperature,
            "model_kwargs": {
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
        }

        if type(model) == OpenAIModels:
            return ChatOpenAI(**_kwargs)
        elif type(model) == TogetherModels:
            return Together(
                together_api_key=Config.TOGETHER_API_KEY,
                verbose=True,
                **{k: v for k, v in _kwargs.items() if k not in ("model_kwargs", "n")}
                # model="togethercomputer/RedPajama-INCITE-7B-Base",
                # temperature=0.7,
                # max_tokens=128,
                # top_k=1,
                # together_api_key="..."
            )
        else:
            raise NotImplementedError(f"LLM for {type(model)} is not implemented.")
