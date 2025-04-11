import unittest

from transformers.agents.tools import add_description

from gptfuzzer.llm.llm import LocalTransformersLLM

class LocalTransformersLLMTest(unittest.TestCase):

    @add_description("Genera dos respuestas llamado al modelo llama-3.2-1b-Instruct")
    def test_generate_response(self):
        llm = LocalTransformersLLM("meta-llama/Llama-3.2-1B-Instruct", "You are a helpful assistant. You will answer in spanish.")

        response = llm.generate("[INST]Which are the most common use cases for the artificial intelligence?[/INST]",
                       0.1,
                       512,
                       2,
                       1,
                       5)
        self.assertEqual(len(response), 2)


if __name__ == '__main__':
    unittest.main()
