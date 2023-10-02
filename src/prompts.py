from abc import ABC, abstractmethod
from typing import Dict


class PromptTreatment(ABC):
    def __init__(self, treatment: str, formatting_map: Dict[str, str] = None) -> None:
        self.treatment = treatment
        self.formatting_map = formatting_map

    def __str__(self) -> str:
        prompt = getattr(self, self.treatment, self.default)
        return str(prompt).replace("\t", "")

    @property
    def formatted_string(self):
        return str(self).format(**self.formatting_map)

    @property
    @abstractmethod
    def default(self):
        pass


class GamePrompt(PromptTreatment):
    default = """
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


class InitialGuesserPrompt(PromptTreatment):
    default = """
        {game_prompt}

        # Hints
        Below are some data structures to help you deduce the card.
        Values: {values}
        Suits: {suits}

        I am the judge; ask for your first hint.
    """


class InitialJudgePrompt(PromptTreatment):
    default = """
        {game_prompt}

        You will never lie.
        You will only respond to guesses and hints.
        When the guesser identifies the card, you will respond with "EOF"

        The card you picked from the deck is {card}
    """
