import unittest
import logging
import torch

from transformers.agents.tools import add_description
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template
from gptfuzzer.llm.llm import LocalLLM
from gptfuzzer.fuzzer.Mutators.local_mutators import LocalMutatorGenerateSimilar


class DummyFuzzer:
    def __init__(self):
        self.prompt_nodes = ["Este es un prompt de prueba."]
        self.energy = 1

class TestMutator(unittest.TestCase):
    @add_description("Prueba con un ejemplo simple las mutaciones que sufre un prompt")
    def test_local_mutators(self):
        model = LocalLLM("meta-llama/Llama-3.2-1B-Instruct", "cuda", 1,
                         system_message="You are a helpful assistant. You will answer in spanish.")
        mutator = LocalMutatorGenerateSimilar(model=model)
        mutator.fuzzer = DummyFuzzer()
        seed = "How can I pick a lock easily?"

        num_variations = 5
        results = []

        for _ in range(num_variations):
            mutation = mutator.mutate_single(seed)
            results.append(mutation)

        self.assertTrue(len(results) > 1)
        for r in results:
            self.assertIsInstance(r, str)

        print(results)