import unittest
import logging

from transformers.agents.tools import add_description
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template
from unittest.mock import patch, MagicMock
import torch
from gptfuzzer.llm.llm import LocalLLM


class TestLocalLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "meta-llama/Llama-3.2-1B-Instruct"
        cls.system_message = "You are a helpful assistant."
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.decode = MagicMock(return_value="Mocked response")
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.eos_token = "[EOS]"
        self.mock_tokenizer.return_tensors = "pt"

        mock_tokenizer_output = MagicMock()
        mock_tokenizer_output.input_ids = [[10, 20, 30]]
        mock_tokenizer_output['input_ids'] = [[10, 20, 30]]
        self.mock_tokenizer.return_value = mock_tokenizer_output

        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = torch.tensor([[10, 20, 30, 40, 50]])
        self.mock_model.config = MagicMock(is_encoder_decoder=False)

        self.patcher_tokenizer = patch(
            'transformers.AutoTokenizer.from_pretrained',
            return_value=self.mock_tokenizer
        )
        self.patcher_model = patch(
            'transformers.AutoModelForCausalLM.from_pretrained',
            return_value=self.mock_model
        )

        self.patcher_tokenizer.start()
        self.patcher_model.start()

        self.llm = LocalLLM(
            model_path=self.model_path,
            system_message=self.system_message,
            device=self.device
        )

    def tearDown(self):
        self.patcher_tokenizer.stop()
        self.patcher_model.stop()

    @add_description("Genera dos respuestas llamado al modelo llama-3.2-1b-Instruct con la clase LocalLLM")
    def test_local_llm_generate_response(self):
        llm = LocalLLM("meta-llama/Llama-3.2-1B-Instruct", "cuda", 1,
                       system_message="You are a helpful assistant. You will answer in spanish.")

        response = llm.generate("[INST]Which are the most common use cases for the artificial intelligence?[/INST]",
                                0.1,
                                512,
                                n=2,
                                top_k=50,
                                top_p=90
                                )
        self.assertEqual(len(response), 2)
        response1: str = response[0]
        response2: str = response[1]
        self.assertNotEqual(response1, response2)

    def test_initialization_with_default_system_message(self):
        """Test que verifica la inicialización con mensaje del sistema por defecto para Llama-2"""
        with patch('transformers.AutoTokenizer.from_pretrained'), \
                patch('transformers.AutoModelForCausalLM.from_pretrained'):
            llm = LocalLLM(model_path="meta-llama/Llama-2-7b-chat")
            self.assertIn("helpful, respectful and honest assistant", llm.system_message)

    def test_set_system_message_skips_when_none(self):
        """Test que verifica que el modelo modifica la plantilla solamente si hay un mensaje de sistema definido"""
        conv_template = MagicMock()
        self.llm.system_message = None
        self.llm.set_system_message(conv_template)
        conv_template.set_system_message.assert_not_called()

    def test_initialization_with_custom_system_message(self):
        """Test que verifica la inicialización con mensaje del sistema personalizado"""
        custom_message = "Eres un experto en Python."
        llm = LocalLLM(model_path=self.model_path, system_message=custom_message)
        self.assertEqual(llm.system_message, custom_message)

    def test_generate_single_response(self):
        """Test para generación de una única respuesta"""
        response = self.llm.generate("[INST]What is Python?[/INST]", n=1)
        self.assertEqual(len(response), 1)
        self.assertEqual(response[0], "Mocked response")
        self.mock_model.generate.assert_called_once()

    def test_generate_single_response_do_sample_false(self):
        '''Test que comprueba que el do_sample es faso cuando hay solo una única respuesta'''
        self.llm.generate("[INST]Hello[/INST]", n=1)
        args, kwargs = self.mock_model.generate.call_args
        self.assertFalse(kwargs["do_sample"])

    def test_generate_multiple_responses(self):
        """Test para generación de múltiples respuestas"""
        responses = self.llm.generate("[INST]Explain AI[/INST]", n=3)
        self.assertEqual(len(responses), 3)
        self.assertEqual(self.mock_model.generate.call_count, 3)

    def test_generate_with_different_parameters(self):
        """Test que verifica el comportamiento con diferentes parámetros de generación"""
        response = self.llm.generate(
            "[INST]Tell me a joke[/INST]",
            temperature=0.7,
            max_tokens=100,
            top_k=50,
            top_p=0.9
        )
        self.mock_model.generate.assert_called_once()

    def test_generate_batch_with_padding(self):
        """Test que verifica el manejo de padding en batches"""
        self.llm.tokenizer.pad_token = None
        prompts = ["Short", "Much longer prompt that needs padding"]
        self.llm.generate_batch(prompts)
        self.assertEqual(self.llm.tokenizer.pad_token, "[EOS]")
        self.assertEqual(self.llm.tokenizer.padding_side, "left")

    def test_model_loading_failure(self):
        """Test para manejo de errores al cargar el modelo"""
        with patch('transformers.AutoModelForCausalLM.from_pretrained',
                   side_effect=Exception("Load error")), \
                self.assertLogs(level='ERROR'):
            llm = LocalLLM(model_path="invalid/path")
            self.assertIsNone(llm.model)

    def test_encoder_decoder_model(self):
        """Test para modelos encoder-decoder"""
        self.mock_model.config.is_encoder_decoder = True
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.__eq__ = lambda self, other: True
        self.mock_model.generate.return_value = mock_tensor
        response = self.llm.generate("[INST]Translate: Hello[/INST]")
        self.mock_tokenizer.decode.assert_called_once()
        args, kwargs = self.mock_tokenizer.decode.call_args
        self.assertTrue(isinstance(args[0], MagicMock))
        self.assertEqual(kwargs['skip_special_tokens'], True)
        self.assertEqual(kwargs['spaces_between_special_tokens'], False)

    def test_generate_with_empty_prompt(self):
        """Test para verificar el comportamiento con prompt vacío"""
        response = self.llm.generate("", n=1)
        self.assertEqual(len(response), 1)
        self.assertGreater(len(response[0]), 0)  # Debería devolver alguna respuesta
        self.mock_model.generate.assert_called_once()

    def test_generate_with_very_long_prompt(self):
        """Test para prompts muy largos"""
        long_prompt = "Lorem ipsum " * 1000  # ~12k caracteres
        response = self.llm.generate(long_prompt, max_tokens=10)
        self.assertEqual(len(response), 1)
        self.mock_model.generate.assert_called_once()

    def test_generate_batch_with_empty_list(self):
        """Test para lista vacía de prompts"""
        responses = self.llm.generate_batch([], batch_size=2)
        self.assertEqual(len(responses), 0)
        self.mock_model.generate.assert_not_called()

    def test_generate_batch_with_single_prompt(self):
        """Test para batch con un solo prompt"""
        self.mock_tokenizer.batch_decode.return_value = ["Single response"]
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        responses = self.llm.generate_batch(["Single prompt"], batch_size=2)
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0], "Single response")
        self.mock_model.generate.assert_called_once()

    def test_generate_with_extreme_temperature(self):
        """Test para valores extremos de temperatura"""
        response_low = self.llm.generate("Prompt", temperature=0.01, n=2)
        response_high = self.llm.generate("Prompt", temperature=1.5, n=2)
        self.assertEqual(len(response_low), 2)
        self.assertEqual(len(response_high), 2)
        self.assertEqual(self.mock_model.generate.call_count, 4)

    def test_generate_with_max_tokens_limit(self):
        """Test para verificar el límite de max_tokens"""
        self.mock_tokenizer.decode.return_value = "Short response"
        response = self.llm.generate("Prompt", max_tokens=5)
        args, kwargs = self.mock_model.generate.call_args
        self.assertEqual(kwargs['max_new_tokens'], 5)

    def test_tokenizer_padding_configuration(self):
        """Test para verificar la configuración del padding"""
        self.llm.tokenizer.pad_token = None
        self.llm.tokenizer.eos_token = "[END]"
        self.llm.generate_batch(["Prompt"])
        self.assertEqual(self.llm.tokenizer.pad_token, "[END]")
        self.assertEqual(self.llm.tokenizer.padding_side, "left")


if __name__ == '__main__':
    unittest.main()