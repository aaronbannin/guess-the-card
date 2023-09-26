# Evaluating ChatGPT's Ability to Reason
Does ChatGPT have the ability to reason about novel problems, or is it limited to simply generating pre-learned text? There are some really fascinating examples of assessing ChatGPT's reasoning capabilities, often focused on mathematics. These assessments focus on existing concepts which are most likely within the training set. Testing GPT against a novel problem will assess its ability to learn and allows us to measure its ability to reason.

# Let's Play a Game
To assess the reasoning capability of GPT, I have two agents play a game with each other. The initial prompt outlines the rules for each agent. Summarized, it's a two player game where the guesser must deduce the card that is held by the judge. There are a few important design elements of this game:
- **Turn Based**: Players take actions in sequence. This allows for asynchronous play and is conducive to scripting.
- **Asymmetric Information**: The judge knows that value of the card, while the guesser does not.
- **Winnable**: It is possible, and arguable rather easy, for the guesser to win every round.

A player is defined as an instance of a [LangChain Chain](https://python.langchain.com/docs/modules/chains/). Both players are primed with a prompt explaining the rules and assigning their role. Langchain also manages each agent's [memory](https://python.langchain.com/docs/modules/memory/), which is used to populate the context window on every interaction.

It is important to note that ChatGPT is probablisitc, not deterministic. The same prompt will generate different results, and each response impacts future responses. This randomness is controlled primarily by the judge, whose instructions minimizes the scope of possible responses.

ChatGPT exhibits a noticeable bias towards positivity and helpfulness, leading to some intriguing patterns of behavior. Given the extensive attention GPT's problematic behavior has garnered, it's my belief that OpenAI has overcompensated with a strong bias towards positivity. As a result, some interactions with the model seem distinctly non-human. Detailed examples are elaborated upon below.


# Observations
- **[Grammatical Person](https://en.wikipedia.org/wiki/Grammatical_person)**: Agents had difficulty speaking in the first person ("I will ..." or "Is the card ..."), often favoring the third person ("The guesser will ..."). This might be because the instructions are written in third person, thus priming the agent to adopt similar language.

    ```
    guesser: Thank you for the hint. With this information, we can narrow down the possible values of the card to king or ace. Now, let me ask the judge for a hint about the suit of the card. I will ask, "Is the suit of the card hearts?"
    judge:   The judge is responding to the hint, "Is the suit of the card hearts?" with "correct."
    ```

- **Role Reversal**: Agents would occasionally invert roles; the judge would start guessing and the guesser would start judging. This seems to be a symptom of the "nice" bias; ChatGPT is trained to always defer to the user.

    ```
    guesser: Thank you for the reminder! I will ask the judge for a hint about the value of the card. Is the value of the card higher than 7?
    judge:   The value of the card is lower than 7.
    guesser: Thank you for the hint! Based on your response, I can now narrow down the possible values of the card to 1, 2, 3, 4, 5, 6, and 7. What is the next hint you would like to ask the judge?
    ```

    ```
    guesser: As the guesser, I would like to ask the judge for a hint about the value of the card.
    judge:   Sure! Please go ahead and ask for a hint about the value of the card.
    guesser: Is the value of the card between 2 and 6 (inclusive)?
    judge:   Yes, the value of the card is between 2 and 6 (inclusive).
    guesser: Great! Now we know that the value of the card is between 2 and 6 (inclusive). What hint would you like to ask the judge next?
    judge:   Is the value of the card higher than 4?
    guesser: Yes, the value of the card is higher than 4.
    ```

- **Ending the game**: ChatGPT will always respond to a prompt. For this game, a simple `while` loop is used to engage each player; this means that the conversation will continue ad infinitum. Ideally, the game would conclude when the win condition is met. The judge is instructed to respond with `EOF` when the game is complete, similar in concept to the `<|endoftext|>` token used by GPT to terminate it's output. In practice, most conversations were terminated by capping the number of rounds played. This is because the guesser typically failed to win the game, and occassionally the judge simply forgot to use `EOF` in their response.


- **Memory**: The biggest boost in judge performance came from customizing LangChain's memory manager. The `ConversationBufferMemory` keeps a log of the agent's conversation and submits the log to the LLM on each prompt. In the game, the judge is stateless; it does not need to remember any actions taken by the guesser. To respond, the judge just needs the rules for the game and the most recent prompt from the guesser. However, the guesser does need conversation history, as it needs to remember all hints and guesses taken within the game to strategize it's next guess.

    The initial implementation of memory kept a full log of the conversation for each player. The players generally followed the concept of the rules of the game; the concept of responses were limited to hints, guesses, and verifying the card. However, the players took liberties in how their responses were phrased. The judge's behavior became much more reliable when the prompt only includes the game's rules and the guesser's most recent response. Notably, the judge's responses kept in line with the game rules. This suggests information entropy within the context window; agents are able to maintain their path of reasoning but may forget detailed instructions.

- **Auditing**: Manually reviewing and labeling conversations is a labor-intensive process. To expedite this, I attempted to automate the labeling to quickly gauge modifications in prompts. I designed a prompt aimed at evaluating a conversation to determine the guesser's victory in the game. Although the prompt effectively summarizes the game's progress, it's prone to producing false positives. GPT's evaluations are heavily swayed by the guesser's feedback. For instance, if the guesser claims a win, GPT's audit often concurs. Such imprecision in GPT's roles as both guesser and judge impairs its capability to accurately assess chat transcripts.


# Next Steps:
- Provide guesser with guessing strategies.
- Prime guesser with techniques of remembering learned information.
- Fine-tune model to automate auditing and labeling.
- Rephrase system prompt to encourage first-person responses.
- Experiment with functions; might make it easier to parse responses.
- Loosen response guidance and allow players to explore different ways of phrasing hints and guesses.
- Test against other models.

# Run Locally
- Verify that your Python version matches the version set in `pyproject.toml`.
- This tool requires a PostgreSQL database to manage data. On a Mac, I find [Postgres.app](https://postgresapp.com/) to be the easiest way to get up and running. You may need to add this line to your `~/.zshrc` file:

    ```export PATH="/Applications/Postgres.app/Contents/Versions/latest/bin:$PATH"```

    [Homebrew](https://wiki.postgresql.org/wiki/Homebrew) is another effective way to install.

- Create an `.env` file by running `cp .env.example .env` and update the values. Refer to `config.py` for environment variables.
- Install and set up [Poetry](https://python-poetry.org/docs/#installation).
- Run `poetry install`.
- Run `poetry run python cli.py --help` to see availible commands.
