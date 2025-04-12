import unittest

from transformers.agents.tools import add_description

from gptfuzzer.llm.llm import LocalLLM

class LlmTest(unittest.TestCase):

    @add_description("Genera dos respuestas llamado al modelo llama-3.2-1b-Instruct con la clase LocalLLM")
    def test_local_llm_generate_response(self):
        llm = LocalLLM("meta-llama/Llama-3.2-1B-Instruct", "cuda", 1, system_message="You are a helpful assistant. You will answer in spanish.")

        response = llm.generate("[INST]Which are the most common use cases for the artificial intelligence?[/INST]",
                       0.1,
                       512,
                        n = 2,
                        top_k= 50,
                        top_p= 90
                        )
        self.assertEqual(len(response), 2)
        response1: str = response[0]
        response2: str = response[1]
        self.assertNotEqual(response1, response2)


if __name__ == '__main__':
    unittest.main()
