from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid1, UUID

import click
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from sqlalchemy.orm.session import Session

from postgres import postgres_engine, ChatLogs


load_dotenv()

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

        click.echo(f"{self.role.name}: {str(response).strip()}")
        return str(response)

game_prompt = """
You are playing a game called 'Guess the card'.
There are two players: the guesser and the judge.

We start with a standard poker deck of 52 cards where the lowest value is 2 and the highest value is A.
The suits, ordered left to right, are: diamonds, hearts, clubs, spades. Thus, clubs is to the right of both diamonds and hearts.
The judge will randomly pick a card from the deck.

The guesser will ask for hints from the judge until they are ready to guess the card.
The guesser can ask for a hint in the form of 'Is the value of the card <value>?' or 'Is the suit of the card <suit>?'
If the hint is for the value, the judge will respond 'higher', 'lower', or 'correct'.
If the hint is for the suit, the judge will respond 'left', 'right', or 'correct'.

The guesser can guess the card with the statement 'The card is a <value> of <suit>'.
If the guesser is incorrect, the current card is discarded and the judge draws a new card.
The guesser must guess the value in the correct format to win.

The judge will never lie.
The judge will only respond to guesses and hints. The judge will respond with 'I did not understand, would you like to make a guess or a hint?'
When the game ends, the judge will respond with "EOF"
You will be the {role}.
"""

audit_prompt = """
{log}

You are auditing the result of a conversation. The guesser or judge may lie or make a mistake.
The guesser must guess the card with the statement 'The card is a <value> of <suit>'.
What is the exact suit and exact value of the card?
Did the guesser win the game by correctly guessing the suit and value of the card?
"""


@click.group()
def cli():
    pass

@cli.command()
def play():
    run = Run()
    click.echo(f"Run {run.id}")
    session = Session(postgres_engine)
    click.echo(f"game_prompt {game_prompt}")

    llm = ChatOpenAI(
        max_tokens=256,
        model="gpt-3.5-turbo-0613",
        n=1,
        temperature=1.0,
        model_kwargs={
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    )

    judge = Agent(run=run, role=Role.judge, llm=llm, session=session)
    guessor = Agent(run=run, role=Role.guesser, llm=llm, session=session)

    # prime the judge
    judge.send_chat_message(game_prompt.format(role="judge") + "\nRespond with 'EOF' when the guesser has won the game.")
    # prime the guesser; output will mutate in the loop below
    guesser_response = guessor.send_chat_message(game_prompt.format(role="guessor") + "\nAsk for a hint from the judge.")

    iterations = 0
    while True:
        judge_loop = judge.send_chat_message(guesser_response)
        guesser_response = guessor.send_chat_message(judge_loop)

        iterations += 1
        eof = "EOF" in judge_loop
        if (
            # "EOF" in guesser_response or
            eof or
            iterations > 15
        ):
            click.echo(f"Ending condition met EOF {eof} iterations {iterations}")
            break

    session.close()

@cli.command()
@click.option('--run-id', '-r', help='run_id of game to be audited', type=click.STRING)
def audit(run_id: click.STRING):
    session = Session(postgres_engine)
    results = session.query(ChatLogs).filter(ChatLogs.run_id == run_id).order_by(ChatLogs.created_at)

    replayed_log = [f"{result.role}: {result.response}" for result in results]

    llm = ChatOpenAI(
        max_tokens=256,
        model="gpt-3.5-turbo-0613"
    )
    audit_chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())
    response = audit_chain.run(input=audit_prompt.format(log=replayed_log))
    click.echo(f"Audit response: {response}")

if __name__ == '__main__':
    cli()
