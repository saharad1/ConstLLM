from typing import Optional, cast

import torch
from torch.nn import DataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


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

        model_kwargs = {"device_map": f"{self.device}"}
        # model_kwargs = {"device_map": "auto"}
        if self.use_quantization:
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
            model_kwargs["torch_dtype"] = torch.float16  # Use full precision (or float16 if preferred)

        # if self.use_quantization:
        #     quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_compute_dtype=torch.float16, # torch.float16 (match with the model's dtype),
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_quant_type="nf4"
        #     )
        #
        #     # quantization_config = BitsAndBytesConfig(
        #     #     load_in_8bit=True,
        #     #     # bnb_8bit_compute_dtype=torch.float16,
        #     # )
        #
        #     # Define the quantization configuration for 8-bit
        #     # quantization_config = BitsAndBytesConfig(
        #     #     load_in_8bit=True,  # Enable 8-bit quantization
        #     #     bnb_8bit_compute_dtype=torch.bfloat16,  # Use FP16 for computations (optional)
        #     #     bnb_8bit_use_double_quant=False,  # Disable double quantization for stability
        #     #     bnb_8bit_quant_type="fp8"  # Use FP8 quantization type (for more stability)
        #     # )
        #     model_kwargs["quantization_config"] = quantization_config
        # else:
        #     model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        # Set up padding token after model initialization
        self._setup_padding_token()

        # # Apply DataParallel if enabled
        # if self.use_dataparallel:
        #     if not self.device_ids:
        #         self.device_ids = list(range(torch.cuda.device_count()))  # Use all available GPUs
        #     self.model = DataParallel(self.model, device_ids=self.device_ids)
        #     print(f"Model wrapped in DataParallel on GPUs: {self.device_ids}")
        # else:
        #     print(f"Model loaded on device: {self.device}")

        model_device = next(self.model.parameters()).device
        print(f"Model loaded on device: {model_device}")
        if self.use_quantization:
            print("Model quantized to 4-bit precision")
        else:
            print("Model loaded without quantization")

    def _setup_padding_token(self) -> None:
        """
        Ensure that the tokenizer has a padding token set.

        If no padding token is set, this method adds a custom padding token
        and resizes the model's token embeddings accordingly.
        Uses '[PAD]' specifically for Mistral models.
        """
        if self.tokenizer.pad_token is None:
            # Check if it's a Mistral model and use [PAD] specifically for Mistral
            is_mistral = "mistral" in self.model_id.lower()
            pad_token = "[PAD]" if is_mistral else "<|pad|>"

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
