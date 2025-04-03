from typing import Optional, cast

import torch
from torch.nn import DataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from unsloth import FastLanguageModel


class ModelTokenizerBundle:
    """
    A bundle class that encapsulates a pre-trained language model, its associated tokenizer,
    and the device on which the model is loaded.

    This class simplifies the process of initializing and managing the components
    necessary for natural language processing tasks using transformer models.

    Attributes:
    model_id (str): The identifier of the pre-trained model.
    tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
    model (PreTrainedModel): The pre-trained language model.
    device (torch.device): The device on which the model is loaded.
    use_quantization (bool): Whether to use 4-bit quantization for the model.
    """

    def __init__(
        self,
        model_id: str,
        device="cuda",
        use_quantization: bool = True,
        quantization_type: Optional[str] = "4bit",
    ):
        """
        Initialize the ModelTokenizerBundle with a specified model ID and quantization option.

        Args:
            model_id (str): The identifier of the pre-trained model to load.
            use_quantization (bool, optional): Whether to use 4-bit quantization. Defaults to True.
        """
        self.model_id: str = model_id
        self.use_quantization: bool = use_quantization
        self.quantization_type: Optional[str] = quantization_type
        self.tokenizer = None
        self.model = None
        self.device = device

        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize the model, tokenizer, and device with optional quantization.

        This method sets up the tokenizer, optionally configures quantization, loads the model,
        and determines the appropriate device for the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Check if model is already quantized based on its name
        is_already_quantized = any(qt in self.model_id.lower() for qt in ["bnb-4bit", "4bit", "bnb-8bit", "8bit"])

        # Check if this is an unsloth model
        is_unsloth_model = "unsloth" in self.model_id.lower()

        if is_unsloth_model:
            # For unsloth models, we need to use their specialized loading
            print(f"Detected Unsloth model: {self.model_id}, using Unsloth-specific loading")

            try:

                print(f"Loading Unsloth model: {self.model_id}")

                # Use unsloth's specialized model loading
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_id,
                    max_seq_length=2048,  # Adjust as needed for your use case
                    dtype=torch.float16,
                    load_in_4bit=False,  # Disable 4-bit loading for now
                    device_map=self.device,
                )
                print(f"Unsloth model loaded successfully on device: {self.device}")
            except ImportError as e:
                print(f"Required package not found: {str(e)}")
                raise
        else:
            # Standard model loading for non-unsloth models
            model_kwargs = {"device_map": f"{self.device}"}

            # Only apply quantization if explicitly requested AND model is not already quantized
            if self.use_quantization and not is_already_quantized:
                if self.quantization_type == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                elif self.quantization_type == "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.bfloat16,
                    )
                else:
                    raise ValueError(f"Unsupported quantization type: {self.quantization_type}")
                model_kwargs["quantization_config"] = quantization_config
            else:
                if is_already_quantized:
                    print(f"Model '{self.model_id}' appears to be already quantized. Skipping additional quantization.")
            model_kwargs["torch_dtype"] = torch.float16  # Use full precision (or float16 if preferred)

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        # Set up padding token after model initialization
        self._setup_padding_token()

        model_device = next(self.model.parameters()).device
        print(f"Model loaded on device: {model_device}")
        if not is_unsloth_model and self.use_quantization and not is_already_quantized:
            print("Model quantized to 4-bit precision")
        else:
            print("Model loaded without quantization")

    def _setup_padding_token(self) -> None:
        """
        Ensure that the tokenizer has a padding token set.

        If no padding token is set, this method adds a custom padding token
        and resizes the model's token embeddings accordingly.
        Uses model-specific padding tokens:
        - '[PAD]' for Mistral models
        - '<pad>' for Qwen models
        - ' ' for other models
        """
        if self.tokenizer.pad_token is None:
            # Check model type and set appropriate padding token
            is_mistral = "mistral" in self.model_id.lower()
            is_qwen = "qwen" in self.model_id.lower()

            # Set appropriate padding token based on model type
            if is_mistral:
                pad_token = "[PAD]"
            elif is_qwen:
                pad_token = "<pad>"
            else:
                pad_token = ""

            # Add the appropriate padding token
            if pad_token not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.padding_side = "left"
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            print(f"Padding token added: {pad_token}")
        else:
            print(f"Padding token already exists: {self.tokenizer.pad_token}")

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create an attention mask for the given input IDs.

        Args:
            input_ids (torch.Tensor): The input IDs tensor.

        Returns:
            torch.Tensor: The attention mask tensor.
        """
        return (input_ids != self.tokenizer.pad_token_id).long().to(self.device)

    def __str__(self) -> str:
        """
        Return a string representation of the ModelTokenizerBundle.

        Returns:
            str: A string describing the ModelTokenizerBundle instance.
        """
        quantization_status = "with" if self.use_quantization else "without"
        return f"ModelTokenizerBundle(model_id={self.model_id}, device={self.device}, {quantization_status} quantization)"
