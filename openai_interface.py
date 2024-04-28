import abc
import os
from typing import List, Tuple
import openai
import hashlib
import logging
import pickle

import tiktoken
from transformers import GPT2TokenizerFast

from settings import LLMConfig as lc

class OpenAI_Model:
    def __init__(self):
        assert "OPENAI_API_KEY" in os.environ
        self._openai = openai.OpenAI()

        self._model = lc.model
        self._cache_dir = lc.cache_dir
        if not os.path.exists(self._cache_dir) or not os.path.isdir(self._cache_dir):
            os.makedirs(self._cache_dir, exist_ok=True)
        self._max_tokens = lc.max_tokens
        if "gpt-4" in self._model or "gpt-3.5" in self._model:
            # self._tokenizer = tiktoken.get_encoding("cl100k_base")
            self._tokenizer = tiktoken.encoding_for_model(self._model)

        elif "davinci" in self._model:
            self._tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/text-davinci-003')
        else:
            # default
            self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            

    def sample_completions(self, conversation, temperature, seed, num_completions, disable_cache=False) -> Tuple[List[str], str]:
        """Cached LLM query.

        Args:
            conversation (list[dict]): list of messages
            temperature (float): 
            seed (int): 
            num_completions (int): 

        Returns:
            list[str]: responses from the LLM
        """
        prompts_id = ""
        for prompt in conversation:
            prompts_id += str_to_identifier(prompt["content"])
        config_id = f"{temperature}_{seed}_{num_completions}"
        cache_filename = f"{prompts_id}_{config_id}.pkl"
        if len(cache_filename) > 255:
            # Truncate the prompt id
            length = 255 - len(config_id) - 5
            prompts_id = prompts_id[len(prompts_id) - length:]
            cache_filename = f"{prompts_id}_{config_id}.pkl"
        cache_filepath = os.path.join(self._cache_dir, cache_filename)
        if not os.path.exists(cache_filepath) or disable_cache:
            completions = self._sample_completions(conversation, temperature, seed, num_completions)
            with open(cache_filepath, 'wb') as f:
                pickle.dump(completions, f)
            # print(f"Saved to {cache_filepath}")
        else:
            with open(cache_filepath, 'rb') as f:
                # print("Cache hit", cache_filepath)
                completions = pickle.load(f)
        return completions, cache_filepath

    def _sample_completions(self, conversation, temperature, seed, num_completions) -> List[str]:
        """Query the LLM.

        Returns:
            list[str]: responses from the LLM

        """
        num_prompt_tokens = sum([len(self._tokenizer.encode(prompt["content"])) for prompt in conversation])
        max_response_tokens = self._max_tokens - num_prompt_tokens
        if max_response_tokens <= 0:
            logging.warn("Max tokens exceeded by prompts")
            return []

        completion = self._openai.chat.completions.create(
            model=self._model,
            messages = conversation,
            max_tokens=max_response_tokens,
            temperature=temperature,
            n=num_completions,
            seed=seed,
        )
        responses = [c.message.content for c in completion.choices]
        return responses

def str_to_identifier(x: str) -> str:
    """Convert a string to a small string with negligible collision probability
    and where the smaller string can be used to identifier the larger string in
    file names.

    Importantly, this function is deterministic between runs and between
    platforms, unlike python's built-in hash function.

    References:
        https://stackoverflow.com/questions/45015180
        https://stackoverflow.com/questions/5297448

    TODO: this sometimes hashes the same string to different smaller strings.
    """
    return hashlib.md5(x.encode('utf-8')).hexdigest()

