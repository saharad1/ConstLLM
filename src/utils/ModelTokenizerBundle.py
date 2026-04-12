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
        device_map: Optional[str] = None,
        use_quantization: bool = True,
        quantization_type: Optional[str] = "4bit",
    ):
        """
        Initialize the ModelTokenizerBundle with a specified model ID and device configuration.

        Args:
            model_id (str): The identifier of the pre-trained model to load.
            device_map (str, optional): Device mapping strategy. Options:
                - None: Use single GPU (defaults to "cuda:0")
                - "auto": Automatically distribute across available GPUs
                - "balanced": Balance memory across GPUs
                - "sequential": Load layers sequentially across GPUs
                - "cuda:0": Single GPU
                - "cuda:0,cuda:1": Specific GPU mapping
            use_quantization (bool, optional): Whether to use 4-bit quantization. Defaults to True.
            quantization_type (str, optional): Quantization type ("4bit" or "8bit"). Defaults to "4bit".
        """
        self.model_id: str = model_id
        self.device_map: Optional[str] = device_map
        self.use_quantization: bool = use_quantization
        self.quantization_type: Optional[str] = quantization_type
        self.tokenizer = None
        self.model = None

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

                # Determine device mapping for Unsloth models
                device_map = self.device_map if self.device_map is not None else "cuda:0"
                print(f"Using device map for Unsloth: {device_map}")

                from unsloth import FastLanguageModel

                # Use unsloth's specialized model loading
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_id,
                    max_seq_length=2048,  # Adjust as needed for your use case
                    dtype=torch.float16,
                    load_in_4bit=False,  # Disable 4-bit loading for now
                    device_map=device_map,
                )
                print(f"Unsloth model loaded successfully with device map: {device_map}")
            except ImportError as e:
                print(f"Required package not found: {str(e)}")
                raise
        else:
            # Standard model loading for non-unsloth models
            # Determine device mapping strategy
            device_map = self.device_map if self.device_map is not None else "cuda:0"
            model_kwargs = {"device_map": device_map}
            print(f"Using device map: {device_map}")

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

            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            except RuntimeError as e:
                if "cudaGetDeviceCount" in str(e) or "invalid device ordinal" in str(e):
                    print(f"Direct GPU loading failed ({e}). Falling back to CPU load + GPU transfer...")
                    cpu_kwargs = {k: v for k, v in model_kwargs.items() if k != "device_map"}
                    cpu_kwargs["device_map"] = "cpu"
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **cpu_kwargs)
                    if device_map and device_map != "cpu":
                        target_device = device_map if device_map.startswith("cuda") else "cuda:0"
                        print(f"Moving model to {target_device}...")
                        self.model = self.model.to(target_device)
                else:
                    raise

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
        # Create attention mask
        mask = (input_ids != self.tokenizer.pad_token_id).long()

        # For multi-GPU models with device_map, let the model handle device placement
        # For single GPU models, move to the appropriate device
        if self.device_map is None or self.device_map == "cuda:0":
            # Single GPU case - move to device
            mask = mask.to("cuda:0")

        return mask

    def __str__(self) -> str:
        """
        Return a string representation of the ModelTokenizerBundle.

        Returns:
            str: A string describing the ModelTokenizerBundle instance.
        """
        quantization_status = "with" if self.use_quantization else "without"
        device_info = f"device_map={self.device_map}" if self.device_map else "device_map=None"
        return f"ModelTokenizerBundle(model_id={self.model_id}, {device_info}, {quantization_status} quantization)"
