from dataclasses import dataclass
from enum import Enum


class Persona(Enum):
    system = "system"
    human = "human"
    assistant = "assistant"

@dataclass
class Message:
    persona: Persona
    content: str

@dataclass
class Memory:
    messages: list[Message]

    def add_message(self, message: Message) -> None:
        pass

@dataclass
class LLMArgs:
    pass

@dataclass
class LLM:
    args: LLMArgs

    def chat(self, message: str):
        pass

@dataclass
class HTTPLLM(LLM):
    pass

@dataclass
class PackageLLM(LLM):
    pass

@dataclass
class Agent:
    llm: LLM
    memory: Memory
