from datetime import datetime
from enum import Enum
from uuid import uuid1

from click import echo
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from sqlalchemy.orm.session import Session

from postgres import ChatLogs


class Role(Enum):
    judge = "judge"
    guesser = "guesser"

class Run:
    def __init__(self) -> None:
        self.id = uuid1()
        self.started_at = datetime.now()

class Agent:
    def __init__(self, run: Run, role: Role, llm: ChatOpenAI, session: Session, verbose = False) -> None:
        self.run = run
        self.role = role
        self.llm = llm
        self.memory = ConversationBufferMemory()
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

        echo(f"{self.role.name}: {str(response).strip()}")
        return str(response)
