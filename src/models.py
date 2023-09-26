from datetime import datetime
from enum import Enum
from random import choice
from types import DynamicClassAttribute
from typing import Any, Dict
from uuid import uuid1, uuid4

from click import echo
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm.session import Session

from config import Config


# connect to db
url = URL.create(
    drivername="postgresql",
    username=Config.POSTGRES_USER,
    host=Config.POSTGRES_HOST,
    port=5432,
    database=Config.POSTGRES_DB
)

postgres_engine = create_engine(url)

# surely this already exists in the OpenAI library...
class OpenAIModels(Enum):
    gpt3 = "gpt-3.5-turbo-0613"
    gpt4 = "gpt-4"

class Role(Enum):
    judge = "judge"
    guesser = "guesser"

    @DynamicClassAttribute
    def pretty(self, suffix: str = " :"):
        target_length = max([len(k) for k in self.__class__.__members__.keys()])
        suffix_with_padding = " " * (target_length - len(self._name_)) + suffix
        return self._name_ + suffix_with_padding


class Run:
    def __init__(self) -> None:
        self.id = uuid1()
        self.started_at = datetime.now()

class Deck:
    values = [str(i) for i in range(2, 11)] + ["jack", "queen", "king", "ace"]
    suits = ["diamonds", "hearts", "clubs", "spades"]

    @classmethod
    def draw_card(cls):
        deck = [f"{value} of {suit}" for value in cls.values for suit in cls.suits]
        return choice(deck)

class Base(DeclarativeBase):
    pass

class ChatLogs(Base):
    __tablename__ = 'chat_logs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(UUID(as_uuid=True), nullable=False)
    run_started_at = Column(DateTime())
    role = Column(String(100), nullable=False)
    llm = Column(JSON)
    response = Column(Text, nullable=False)
    context = Column(Text)
    created_at = Column(DateTime(), default=datetime.now)
    updated_at = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

    def __str__(self) -> str:
        role = Role[self.role]
        return f"{role.pretty} {str(self.response).strip()}"

class JudgeMemory(ConversationBufferMemory):
    ai_prefix: str = "AI"
    human_prefix: str = "Human"
    system_prefix: str = "System"

    def set_context(self, initial_prompt: str) -> None:
        """Seed messages for chat"""
        self.chat_memory.add_message(SystemMessage(content=initial_prompt))

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Override: Do not add new responses, only carry forward system message."""
        pass

class GuesserMemory(ConversationBufferMemory):
    ai_prefix: str = "AI"
    human_prefix: str = "Human"
    system_prefix: str = "System"

    def set_context(self, rules: str, initial_prompt: str) -> None:
        """Seed messages for chat"""
        self.chat_memory.add_message(SystemMessage(content=rules))
        self.chat_memory.add_message(HumanMessage(content=initial_prompt))
        # empty message because so we don't lose the rules
        self.chat_memory.add_message(HumanMessage(content=""))

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Override: Do not add new responses, only carry forward system message."""
        _, output_str = self._get_input_output(inputs, outputs)
        # remove last message
        self.chat_memory.messages.pop()
        self.chat_memory.add_ai_message(output_str)


class Agent:
    def __init__(
            self,
            run: Run,
            role: Role,
            llm: ChatOpenAI,
            session: Session,
            verbose = False,
            memory: ConversationBufferMemory = None
        ) -> None:
        self.run = run
        self.role = role
        self.llm = llm
        self.memory = memory if memory is not None else ConversationBufferMemory()
        self.chain = ConversationChain(llm=llm, memory=self.memory, verbose=verbose)
        self.session = session

    def send_chat_message(self, message: str) -> str:
        """
        Wraps Chain.run() and logs results.

        ConversationChain.memory needs to be a ConversationBufferMemory object
        """
        response = self.chain.run(input=message)

        log = ChatLogs(
            run_id=self.run.id,
            run_started_at=self.run.started_at,
            role=self.role.name,
            llm=self.llm.to_json(),
            response=response,
            context=self.memory.buffer_as_str
        )
        self.session.add(log)
        self.session.flush()
        self.session.commit()

        echo(log)
        return str(response)

Base.metadata.create_all(postgres_engine)
