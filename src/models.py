from datetime import datetime
from enum import Enum
from json import loads
from random import choice
from types import DynamicClassAttribute
from typing import Any, Dict, List
from uuid import uuid1, uuid4

from click import echo
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm.session import Session
from wrapt_timeout_decorator import *

from config import Config


# connect to db
url = URL.create(
    drivername="postgresql",
    username=Config.POSTGRES_USER,
    host=Config.POSTGRES_HOST,
    port=5432,
    database=Config.POSTGRES_DB,
)

postgres_engine = create_engine(url)


AGENT_CHAIN_PROMPT_TEMPLATE = """
Current conversation:
{history}
Human: {input}
AI:
"""


class AgentChain(ConversationChain):
    """
    Subclassed from ConversationChain for minor customizations
    """

    prompt = PromptTemplate(
        input_variables=["history", "input"], template=AGENT_CHAIN_PROMPT_TEMPLATE
    )

    @timeout(10)
    def run(self, input: str) -> Any:
        """
        Call super().run() with a timeout
        OpenAI occassionally hangs?
        """
        return super().run(input=input)


class OpenAIModels(Enum):
    """
    Models availible from OpenAI
    Sadly, this is not availible in their library
    """

    gpt3_5_turbo_0613 = "gpt-3.5-turbo-0613"
    gpt3_5_turbo = "gpt-3.5-turbo"
    gpt3_5_turbo_16k = "gpt-3.5-turbo-16k"
    gpt3_5_ft = "ft:gpt-3.5-turbo-0613:personal::8G9xDV6J"
    gpt4 = "gpt-4"


class Role(Enum):
    judge = "judge"
    guesser = "guesser"
    system = "system"

    @DynamicClassAttribute
    def pretty(self, suffix: str = " :") -> str:
        target_length = max([len(k) for k in self.__class__.__members__.keys()])
        suffix_with_padding = " " * (target_length - len(self._name_)) + suffix
        return self._name_ + suffix_with_padding

    @classmethod
    def players(self) -> List[str]:
        return [role.value for role in Role if role in (Role.judge, Role.guesser)]


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
    """Life cycle events for a Run"""

    __tablename__ = "chat_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(UUID(as_uuid=True), nullable=False)
    run_started_at = Column(DateTime())
    role = Column(String(100), nullable=False)
    card = Column(Text)
    llm = Column(JSON)
    response = Column(Text, nullable=False)
    context = Column(Text)
    treatment = Column(Text)
    created_at = Column(DateTime(), default=datetime.now)
    updated_at = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

    def __str__(self) -> str:
        role = Role[self.role]
        return f"{role.pretty} {str(self.response).strip()}"


class RunLabel(Base):
    """One Run can have many audits"""

    __tablename__ = "run_labels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(UUID(as_uuid=True), nullable=False)
    response = Column(JSONB)
    context = Column(Text)
    model = Column(Text)
    created_at = Column(DateTime(), default=datetime.now)
    updated_at = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

    _model = OpenAIModels.gpt3_5_turbo.value
    _PROMPT = """
        # System
        You are auditing the result of a conversation. The guesser or judge may lie or make a mistake.
        The guesser must guess the card with the statement 'The card is a <value> of <suit>'.

        # Conversation
        {log}

        # Response Format
        Use the following JSON structure for your response. Your response will be used by software, not a human.
        ```
        {{ guesser_won: bool, overview: str }}
        ```

        Did the guesser correctly guess the card to be the {card}?
        """

    @classmethod
    def get_prompt(cls, log: str, card: str) -> str:
        return cls._PROMPT.format(log=log, card=card).replace("\t", "")

    @classmethod
    def from_run_id(cls, session: Session, run_id: str) -> "RunLabel":
        results = (
            session.query(ChatLogs)
            .filter(ChatLogs.run_id == run_id)
            .order_by(ChatLogs.created_at)
        )
        replayed_log = [str(result) for result in results]
        first_chat_log: ChatLogs = results[0]

        llm = ChatOpenAI(max_tokens=256, model=cls._model, verbose=True)
        prompt = RunLabel.get_prompt(log=replayed_log, card=first_chat_log.card)

        audit_chain = AgentChain(llm=llm, memory=ConversationBufferMemory())
        response = audit_chain.run(
            input=RunLabel.get_prompt(log=replayed_log, card=first_chat_log.card)
        )

        try:
            # in case llm does not return json parsable string
            as_json = loads(response)

            audit = cls(
                run_id=run_id, response=as_json, model=cls._model, context=prompt
            )
        except Exception as e:
            print("Unable to build label from response")
            print(response)
            raise e

        session.add(audit)
        session.commit()
        return audit


class Run:
    def __init__(self) -> None:
        self.id = uuid1()
        self.started_at = datetime.now()


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
        """Override: Only add most recent response"""
        _, output_str = self._get_input_output(inputs, outputs)
        # remove last message
        self.chat_memory.messages.pop()
        self.chat_memory.add_ai_message(output_str)


class Agent:
    def __init__(
        self,
        run: Run,
        role: Role,
        card: String,
        llm: ChatOpenAI,
        session: Session,
        verbose=False,
        memory: ConversationBufferMemory = None,
    ) -> None:
        self.run = run
        self.role = role
        self.card = card
        self.llm = llm
        self.memory = memory if memory is not None else ConversationBufferMemory()
        self.chain = AgentChain(llm=llm, memory=self.memory, verbose=verbose)
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
            card=self.card,
            llm=self.llm.to_json(),
            response=response,
            context=self.memory.buffer_as_str,
        )
        self.session.add(log)
        self.session.commit()

        echo(log)
        return str(response)


Base.metadata.create_all(postgres_engine)
