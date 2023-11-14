import json
from enum import Enum
from typing import Any

import together

from click import echo
from pydantic import BaseModel
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo

from config import Config
from models.llms import TogetherModels


class Persona(Enum):
    system = "system"
    human = "human"
    assistant = "assistant"

class Message(BaseModel):
    persona: Persona
    content: str

    @property
    def as_string(self) -> str:
        return f"{self.persona.value}: {self.content}"

class HackedMemory:
    personas = Persona

    def __init__(self, name: str) -> None:
        self.name = name
        self.buffer: list[Message] = []

    @property
    def buffer_as_str(self) -> str:
        return "\n".join([m.as_string for m in self.buffer])

    def _add_message(self, message: Message) -> list[Message]:
        echo(f"adding message [{self.name}] {message.as_string}")
        self.buffer.append(message)
        return self.buffer

    def add_human_message(self, content: str) -> list[Message]:
        persona = self.personas.human
        return self._add_message(Message(persona=persona, content=content))

    def add_assistant_message(self, content: str) -> list[Message]:
        persona = self.personas.assistant
        return self._add_message(Message(persona=persona, content=content))

    def add_system_message(self, content: str) -> list[Message]:
        persona = self.personas.system
        return self._add_message(Message(persona=persona, content=content))

    def set_buffer(self, messages: list[Message]) -> list[Message]:
        """Setter for messages. Allows for changing the default behavior of appending each new message."""
        self.buffer = messages

class LLAMAMemory(HackedMemory):
    """
    Subclassed to conform to LLAMA dialect for prompt formatting.

    Not sure subclassing is the right approach here.
    Maybe a mixin would be better? Or perhaps adding this to the Persona Enum?
    This is also strangly decoupled from the `stop` argument on the LLM model?

    https://replicate.com/blog/how-to-prompt-llama#system-prompts
    https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    https://discuss.huggingface.co/t/trying-to-understand-system-prompts-with-llama-2-and-transformers-interface/59016/4
    """

    human_template = "[INST]\n{prompt}\n[/INST]"
    system_template = "<<SYS>>\n{prompt}\n<</SYS>>"

    @property
    def buffer_as_str(self) -> str:
        return "\n".join([self._format_message(m) for m in self.buffer])

    def _format_message(self, message: Message) -> str:
        if message.persona == Persona.human:
            return self.human_template.format(prompt=message.content)
        elif message.persona == Persona.system:
            return self.system_template.format(prompt=message.content)
        else:
            return message.content

class JudgeMemory(LLAMAMemory):
    def add_inital_message(self, content) -> list[Message]:
        super().add_human_message(content)

    def add_human_message(self, content: str) -> list[Message]:
        self.set_buffer()
        return super().add_human_message(content)
        # return []

    def add_assistant_message(self, content: str) -> list[Message]:
        # return super().add_assistant_message(content)
        return []

    def set_buffer(self) -> list[Message]:
        self.buffer = self.buffer[:1]

class LLMResponse(BaseModel):
    input: str
    output: str
    raw: dict[str, Any]


class TogetherAI:
    def __init__(self):
        together.api_key = Config.TOGETHER_API_KEY
        # self.model = "togethercomputer/RedPajama-INCITE-7B-Instruct"
        self.model = TogetherModels.llama2_7b_chat.value


    @on_exception(expo, RateLimitException, max_tries=8)
    @limits(calls=1, period=5)
    def chat(self, content: str) -> LLMResponse:
        # print("together prompt")
        # print(content + "\n\n")
        output = together.Complete.create(
            # prompt = "<human>: What are Isaac Asimov's Three Laws of Robotics?\n<bot>:",
            prompt = content,
            # model = "togethercomputer/RedPajama-INCITE-7B-Instruct",
            model = self.model,
            max_tokens = 256,
            temperature = 0.8,
            top_k = 60,
            top_p = 0.6,
            repetition_penalty = 1.1,
            stop = ["[INST]", "None", "User:"]
            # stop = ['<human>', '\n\n']
        )

        # return output['prompt'][0]+output['output']['choices'][0]['text']

        response =  LLMResponse(
            input=output['prompt'][0],
            output=output['output']['choices'][0]['text'],
            raw=output
        )

        return response

    def to_json(self):
        return self.__dict__
