from time import sleep

import click
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from sqlalchemy.orm.session import Session

import models


card = models.Deck.draw_card()
game_prompt = """
# Rules
You are playing a game called 'Guess the card'.
There are two players: the guesser and the judge.

We start with a standard poker deck of 52 cards where the lowest value is 2 and the highest value is A.
The suits, ordered left to right, are: diamonds, hearts, clubs, spades. Thus, clubs is to the right of both diamonds and hearts.

The guesser will ask for hints from the judge until they are ready to guess the card.
The guesser can ask for a hint in the form of 'Is the value of the card <value>?' or 'Is the suit of the card <suit>?'
If the hint is for the value, the judge will respond 'higher', 'lower', or 'correct'.
If the hint is for the suit, the judge will respond 'left', 'right', or 'correct'.

The guesser can guess the card with the statement 'The card is a <value> of <suit>'.
The guesser must guess the value in the correct format to win.

You will be the {role}.
"""

initial_guesser_prompt = """
{game_prompt}

# Hints
Below are some data structures to help you deduce the card.
Values: {values}
Suits: {suits}

I am the judge; ask for your first hint.
""".format(
    game_prompt=game_prompt,
    values=",".join(models.Deck.values),
    suits=",".join(models.Deck.suits),
)

initial_judge_prompt = (
    game_prompt.format(role=models.Role.judge.value)
    + """
You will never lie.
You will only respond to guesses and hints.
When the guesser identifies the card, you will respond with "EOF"

The card you picked from the deck is {card}
""".format(
        card=card
    )
)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--iterations",
    "-i",
    "max_iterations",
    default=15,
    help="Max number of iterations to be played",
    type=click.INT,
)
@click.option(
    "--verbose",
    "-v",
    default=False,
    help="Verbose logging from Langchain",
    type=click.BOOL,
)
def play(max_iterations: click.INT, verbose: click.BOOL):
    """
    Let's play a game!!!

    This will have two agents, one guesser and one judge, play a game of Guess the Card
    """

    run = models.Run()
    click.echo(f"Run {run.id}")

    with Session(models.postgres_engine) as session:
        click.echo(f"initial_judge_prompt {initial_judge_prompt}")

        llm = ChatOpenAI(
            max_tokens=256,
            model=models.OpenAIModels.gpt3_5_turbo.value,
            n=1,
            temperature=1.0,
            model_kwargs={
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
        )

        judge_memory = models.JudgeMemory()
        judge_memory.set_context(initial_judge_prompt)
        judge = models.Agent(
            run=run,
            role=models.Role.judge,
            card=card,
            llm=llm,
            session=session,
            memory=judge_memory,
            verbose=verbose,
        )

        guessor_memory = ConversationBufferMemory()
        guessor = models.Agent(
            run=run,
            role=models.Role.guesser,
            card=card,
            llm=llm,
            session=session,
            memory=guessor_memory,
            verbose=verbose,
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
            verb = "did" if eof else "did not"

            if eof or iterations_played >= max_iterations:
                ending_condition = (
                    f"Ending condition met. The judge {verb} end the game. Total"
                    f" iterations played {iterations_played}."
                )
                log = models.ChatLogs(
                    run_id=run.id,
                    run_started_at=run.started_at,
                    role=models.Role.system.name,
                    card=card,
                    llm=llm.to_json(),
                    response=ending_condition,
                )
                session.add(log)
                session.commit()

                click.echo(ending_condition)
                break


@cli.command()
@click.option("--run-id", "-r", help="Audit a single run", type=click.STRING)
@click.option(
    "--new",
    "-n",
    is_flag=True,
    help="Generate labels for all runs that are not yet labeled",
)
@click.option("--all", "-a", is_flag=True, help="Generate labels for all runs")
def label(run_id: click.STRING, new: click.BOOL, all: click.BOOL):
    """Use LLM to label a run"""

    with Session(models.postgres_engine) as session:
        if not (run_id or new or all):
            raise Exception("Must specify an argument")

        if run_id:
            audit = models.RunLabel.from_run_id(session, run_id)
            return audit.response

        query = session.query(models.ChatLogs.run_id.distinct())

        # default logic is for --all
        if new:
            query = query.outerjoin(
                models.RunLabel, models.ChatLogs.run_id == models.RunLabel.run_id
            ).filter(models.RunLabel.run_id == None)

        results = query.all()
        click.echo(f"total results {len(results)}")
        click.echo(results)

        for row in results:
            click.echo(f"Labeling run_id {row[0]}")
            models.RunLabel.from_run_id(session, row[0])
            # crude rate limiting
            sleep(5)

        return f"{len(results)} runs labeled"


@cli.command()
@click.option("--run-id", "-r", help="run_id of game to be replayed", type=click.STRING)
def replay(run_id: click.STRING):
    """Print chat log"""
    with Session(models.postgres_engine) as session:
        logs = (
            session.query(models.ChatLogs)
            .filter(models.ChatLogs.run_id == run_id)
            .order_by(models.ChatLogs.created_at)
        )

        for log in logs:
            click.echo(log)


if __name__ == "__main__":
    cli()
