from time import sleep

import click
from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
from sqlalchemy.orm.session import Session

import models
from models.together import HackedMemory, TogetherAI, Message, LLAMAMemory
from prompts import GamePrompt, InitialGuesserPrompt, InitialJudgePrompt


card = models.Deck.draw_card()

def play_options(function):
    """Decorator to share options for Play commands"""
    function = click.option(
        "--iterations",
        "-i",
        "max_iterations",
        default=15,
        help="Max number of iterations to be played",
        type=click.INT,
    )(function)
    function = click.option(
        "--treatment",
        "-t",
        default="default",
        help="Choose prompt treatment",
        type=click.STRING,
    )(function)
    function = click.option(
        "--verbose",
        "-v",
        default=False,
        help="Verbose logging from Langchain",
        type=click.BOOL,
    )(function)
    return function

def guess_the_card(max_iterations: click.INT, verbose: click.BOOL, treatment: click.STRING):
    run = models.Run()
    click.echo(f"Run {run.id} Treatment {treatment}")

    game_prompt = GamePrompt(treatment)
    initial_guesser_prompt = InitialGuesserPrompt(
        treatment,
        {
            "game_prompt": game_prompt,
            "values": ",".join(models.Deck.values),
            "suits": ",".join(models.Deck.suits),
        },
    )
    initial_judge_prompt = InitialJudgePrompt(
        treatment, {"game_prompt": game_prompt, "card": card}
    )

    with Session(models.postgres_engine) as session:
        click.echo(f"initial_judge_prompt {initial_judge_prompt.formatted_string}")

        # def get_llm(model: models.OpenAIModels):
        #     return ChatOpenAI(
        #         max_tokens=256,
        #         model=model.value,
        #         n=1,
        #         temperature=0.8,
        #         model_kwargs={
        #             "frequency_penalty": 0,
        #             "presence_penalty": 0,
        #         },
        #     )

        agent_kwargs = {
            "run": run,
            "card": card,
            "session": session,
            "verbose": verbose
        }
        # judge_memory = models.JudgeMemory()
        # judge_memory.set_context(initial_judge_prompt.formatted_string)

        judge_memory = LLAMAMemory(models.Role.judge.value)
        judge_memory.add_human_message(initial_judge_prompt.formatted_string)


        judge = models.Agent(
            # run=run,
            role=models.Role.judge,
            # card=card,
            # llm=models.LLMFactory(models.OpenAIModels.gpt3_5_turbo),
            # llm=models.LLMFactory.chat(models.TogetherModels.llama2_7b),
            llm=TogetherAI(),
            # session=session,
            memory=judge_memory,
            # verbose=verbose,
            **agent_kwargs
        )

        guessor_memory = ConversationBufferMemory()
        guessor = models.Agent(
            # run=run,
            role=models.Role.guesser,
            # card=card,
            # llm=models.LLMFactory(models.OpenAIModels.gpt3_5_ft),
            # llm=models.LLMFactory.chat(models.TogetherModels.llama2_7b),
            llm=TogetherAI(),
            # session=session,
            # memory=guessor_memory,
            memory=LLAMAMemory(models.Role.guesser.value),
            # verbose=verbose,
            **agent_kwargs
        )

        iterations_played = 0
        try:
            # prime the guesser; output will mutate in the loop below
            guesser_response = guessor.send_chat_message(
                initial_guesser_prompt.formatted_string
            )

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
                        llm={},
                        treatment=treatment,
                        response=ending_condition,
                    )
                    session.add(log)
                    session.commit()

                    click.echo(ending_condition)
                    break

        except TimeoutError() as e:
            ending_condition = (
                f"Timeout occurred. Total iterations played {iterations_played}. {e}"
            )
            log = models.ChatLogs(
                run_id=run.id,
                run_started_at=run.started_at,
                role=models.Role.system.name,
                card=card,
                llm={},
                response=ending_condition,
            )
            session.add(log)
            session.commit()


@click.group()
def cli():
    pass


@cli.command()
@play_options
def play(max_iterations: click.INT, verbose: click.BOOL, treatment: click.STRING):
    """
    Let's play a game!!!

    This will have two agents, one guesser and one judge, play a game of Guess the Card
    """
    return guess_the_card(max_iterations, verbose, treatment)


@cli.command()
@play_options
@click.option(
    "--games",
    "-g",
    help="Total number of games to be played",
    type=click.INT,
)
def play_many(games: click.INT, max_iterations: click.INT, verbose: click.BOOL, treatment: click.STRING):
    for game in range(games):
        click.echo(f"Starting game {game}")
        guess_the_card(max_iterations, verbose, treatment)


@cli.command()
@click.option("--run-id", "-r", help="Label a single run", type=click.STRING)
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
            sleep(15)

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
