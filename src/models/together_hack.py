import json
from typing import Any, Dict, Iterator, List, Mapping, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import Extra
from langchain.schema import LLMResult
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output import GenerationChunk


def _stream_response_to_generation_chunk(
    stream_response: str,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return GenerationChunk(
        text=parsed_response.get("response", ""), generation_info=generation_info
    )


class _TogetherCommon(BaseLanguageModel):
    base_url: str = "https://api.together.xyz/inference"
    """Base url the model is hosted under."""

    together_api_key: str
    """Together AI API key. Get it here: https://api.together.xyz/settings/api-keys"""

    model: str = "llama2"
    """Model name to use."""

    temperature: Optional[float] = None
    """Model temperature."""
    top_p: Optional[float] = None
    """Used to dynamically adjust the number of choices for each predicted token based
        on the cumulative probabilities. A value of 1 will always yield the same
        output. A temperature less than 1 favors more correctness and is appropriate
        for question answering or summarization. A value greater than 1 introduces more
        randomness in the output.
    """
    top_k: Optional[int] = None
    """Used to limit the number of choices for the next predicted word or token. It
        specifies the maximum number of tokens to consider at each step, based on their
        probability of occurrence. This technique helps to speed up the generation
        process and can improve the quality of the generated text by focusing on the
        most likely options.
    """
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    repetition_penalty: Optional[float] = None
    """A number that controls the diversity of generated text by reducing the
        likelihood of repeated sequences. Higher values decrease repetition.
    """
    logprobs: Optional[int] = None
    """An integer that specifies how many top token log probabilities are included in
        the response for each token generation step.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    # mirostat: Optional[int] = None
    # """Enable Mirostat sampling for controlling perplexity.
    # (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    # mirostat_eta: Optional[float] = None
    # """Influences how quickly the algorithm responds to feedback
    # from the generated text. A lower learning rate will result in
    # slower adjustments, while a higher learning rate will make
    # the algorithm more responsive. (Default: 0.1)"""

    # mirostat_tau: Optional[float] = None
    # """Controls the balance between coherence and diversity
    # of the output. A lower value will result in more focused and
    # coherent text. (Default: 5.0)"""

    # num_ctx: Optional[int] = None
    # """Sets the size of the context window used to generate the
    # next token. (Default: 2048)	"""

    # num_gpu: Optional[int] = None
    # """The number of GPUs to use. On macOS it defaults to 1 to
    # enable metal support, 0 to disable."""

    # num_thread: Optional[int] = None
    # """Sets the number of threads to use during computation.
    # By default, Ollama will detect this for optimal performance.
    # It is recommended to set this value to the number of physical
    # CPU cores your system has (as opposed to the logical number of cores)."""

    # repeat_last_n: Optional[int] = None
    # """Sets how far back for the model to look back to prevent
    # repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    # repeat_penalty: Optional[float] = None
    # """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    # will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    # will be more lenient. (Default: 1.1)"""

    # temperature: Optional[float] = None
    # """The temperature of the model. Increasing the temperature will
    # make the model answer more creatively. (Default: 0.8)"""

    # stop: Optional[List[str]] = None
    # """Sets the stop tokens to use."""

    # tfs_z: Optional[float] = None
    # """Tail free sampling is used to reduce the impact of less probable
    # tokens from the output. A higher value (e.g., 2.0) will reduce the
    # impact more, while a value of 1.0 disables this setting. (default: 1)"""

    # top_k: Optional[int] = None
    # """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    # will give more diverse answers, while a lower value (e.g. 10)
    # will be more conservative. (Default: 40)"""

    # top_p: Optional[int] = None
    # """Works together with top-k. A higher value (e.g., 0.95) will lead
    # to more diverse text, while a lower value (e.g., 0.5) will
    # generate more focused and conservative text. (Default: 0.9)"""

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

    # @property
    # def _identifying_params(self) -> Mapping[str, Any]:
    #     """Get the identifying parameters."""
    #     return {**{"model": self.model}, **self._default_params}

    def _create_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, "stop": stop, **kwargs}
        response = requests.post(
            url=f"{self.base_url}/api/generate/",
            headers={"Content-Type": "application/json"},
            json={"prompt": prompt, **params},
            stream=True,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            optional_detail = response.json().get("error")
            raise ValueError(
                f"Ollama call failed with status code {response.status_code}."
                f" Details: {optional_detail}"
            )
        return response.iter_lines(decode_unicode=True)

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk


# class Ollama(BaseLLM, _OllamaCommon):
#     """Ollama locally runs large language models.

#     To use, follow the instructions at https://ollama.ai/.

#     Example:
#         .. code-block:: python

#             from langchain.llms import Ollama
#             ollama = Ollama(model="llama2")
#     """

#     class Config:
#         """Configuration for this pydantic object."""

#         extra = Extra.forbid

#     @property
#     def _llm_type(self) -> str:
#         """Return type of llm."""
#         return "ollama-llm"

#     def _generate(
#         self,
#         prompts: List[str],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> LLMResult:
#         """Call out to Ollama's generate endpoint.

#         Args:
#             prompt: The prompt to pass into the model.
#             stop: Optional list of stop words to use when generating.

#         Returns:
#             The string generated by the model.

#         Example:
#             .. code-block:: python

#                 response = ollama("Tell me a joke.")
#         """
#         # TODO: add caching here.
#         generations = []
#         for prompt in prompts:
#             final_chunk = super()._stream_with_aggregation(
#                 prompt,
#                 stop=stop,
#                 run_manager=run_manager,
#                 verbose=self.verbose,
#                 **kwargs,
#             )
#             generations.append([final_chunk])
#         return LLMResult(generations=generations)

#     def _stream(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> Iterator[GenerationChunk]:
#         for stream_resp in self._create_stream(prompt, stop, **kwargs):
#             if stream_resp:
#                 chunk = _stream_response_to_generation_chunk(stream_resp)
#                 yield chunk
#                 if run_manager:
#                     run_manager.on_llm_new_token(
#                         chunk.text,
#                         verbose=self.verbose,
#                     )
