from gptfuzzer.fuzzer.core import GPTFuzzer
from gptfuzzer.fuzzer.Mutators.imutator import Mutator
from gptfuzzer.utils.template_functions import shorten, expand, rephrase, cross_over, generate_similar
from gptfuzzer.llm import LocalLLM


class LocalMutatorBase(Mutator):
    def __init__(self,
                 model: 'LocalLLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(fuzzer)

        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens

    def mutate_single(self, seed) -> 'list[str]':
        return self.model.generate(seed, self.temperature, self.max_tokens, self.n)

class LocalMutatorGenerateSimilar(LocalMutatorBase):
    def __init__(self,
                 model: 'LocalLLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, fuzzer)

    def mutate_single(self, seed):
        return super().mutate_single(generate_similar(seed, self.fuzzer.prompt_nodes))

class LocalMutatorCrossOver(LocalMutatorBase):
    def __init__(self,
                 model: 'LocalLLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, fuzzer)

    def mutate_single(self, seed):
        return super().mutate_single(cross_over(seed, self.fuzzer.prompt_nodes))

class LocalMutatorExpand(LocalMutatorBase):
    def __init__(self,
                 model: 'LocalLLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, fuzzer)

    def mutate_single(self, seed):
        return [r + seed for r in super().mutate_single(expand(seed, self.fuzzer.prompt_nodes))]

class LocalMutatorShorten(LocalMutatorBase):
    def __init__(self,
                 model: 'LocalLLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, fuzzer)

    def mutate_single(self, seed):
        return super().mutate_single(shorten(seed, self.fuzzer.prompt_nodes))

class LocalMutatorRephrase(LocalMutatorBase):
    def __init__(self,
                 model: 'LocalLLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 fuzzer: 'GPTFuzzer' = None):
        super().__init__(model, temperature, max_tokens, fuzzer)

    def mutate_single(self, seed):
        return super().mutate_single(rephrase(seed, self.fuzzer.prompt_nodes))