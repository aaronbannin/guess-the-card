from typing import Any, Dict

import click
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from sqlalchemy.orm.session import Session

from models import postgres_engine, Agent, ChatLogs, Deck, OpenAIModels, Role, Run


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
The guesser must guess the value in the correct format to win.

You will be the {role}.
"""

initial_guesser_prompt = """
Below are some data structures to help you deduce the card.
Values: {values}
Suits: {suits}

I am the judge; ask for your first hint.
""".format(values=",".join(Deck.values), suits=",".join(Deck.suits))

initial_judge_prompt = game_prompt.format(role=Role.judge.value) + """
You will never lie.
You will only respond to guesses and hints.
When the game ends, you will respond with "EOF"

The card you picked from the deck is {card}
""".format(card=Deck.draw_card())

audit_prompt = """
{log}

You are auditing the result of a conversation. The guesser or judge may lie or make a mistake.
The guesser must guess the card with the statement 'The card is a <value> of <suit>'.
What is the exact suit and exact value of the card?
Did the guesser win the game by correctly guessing the suit and value of the card?
"""


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


@click.group()
def cli():
    pass

@cli.command()
@click.option('--iterations', '-i', 'max_iterations', default=1, help='Max number of iterations to be played', type=click.INT)
@click.option('--verbose', '-v', default=False, help='Verbose logging from Langchain', type=click.BOOL)
def play(max_iterations: click.INT, verbose: click.BOOL):
    """
    Let's play a game!!!

    This will have two agents, one guesser and one judge, play a game of Guess the Card
    """

    run = Run()
    click.echo(f"Run {run.id}")

    with Session(postgres_engine) as session:
        click.echo(f"initial_judge_prompt {initial_judge_prompt}")

        llm = ChatOpenAI(
            max_tokens=256,
            model=OpenAIModels.gpt3.value,
            n=1,
            temperature=1.0,
            model_kwargs={
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
        )

        judge_memory = JudgeMemory()
        judge_memory.set_context(initial_judge_prompt)
        judge = Agent(
            run=run,
            role=Role.judge,
            llm=llm,
            session=session,
            memory=judge_memory,
            verbose=verbose
        )

        guessor_memory = ConversationBufferMemory()
        guessor = Agent(
            run=run,
            role=Role.guesser,
            llm=llm,
            session=session,
            memory=guessor_memory,
            verbose=verbose
        )

        # prime the judge
        judge.send_chat_message(initial_judge_prompt)
        # prime the guesser; output will mutate in the loop below
        guesser_response = guessor.send_chat_message(initial_guesser_prompt)

        iterations_played = 0
        while True:
            judge_loop = judge.send_chat_message(guesser_response)
            guesser_response = guessor.send_chat_message(judge_loop)

            iterations_played += 1
            eof = "EOF" in judge_loop
            if (
                eof or
                iterations_played >= max_iterations
            ):
                click.echo(f"Ending condition met EOF {eof} iterations played {iterations_played}")
                break


@cli.command()
@click.option('--run-id', '-r', help='run_id of game to be audited', type=click.STRING)
def audit(run_id: click.STRING):
    """Send log to an LLM to review"""
    with Session(postgres_engine) as session:
        results = session.query(ChatLogs).filter(ChatLogs.run_id == run_id).order_by(ChatLogs.created_at)

        replayed_log = [str(result) for result in results]

        llm = ChatOpenAI(
            max_tokens=256,
            model=OpenAIModels.gpt3.value
        )

        audit_chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())
        response = audit_chain.run(input=audit_prompt.format(log=replayed_log))
        click.echo(f"Audit response: {response}")

@cli.command()
@click.option('--run-id', '-r', help='run_id of game to be replayed', type=click.STRING)
def replay(run_id: click.STRING):
    """Print chat log"""
    with Session(postgres_engine) as session:
        logs = session.query(ChatLogs).filter(ChatLogs.run_id == run_id).order_by(ChatLogs.created_at)

        for log in logs:
            click.echo(log)


if __name__ == '__main__':
    cli()
