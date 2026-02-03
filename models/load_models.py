'''
Load the model and tokenizer
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time # Import time for timing
from utils.utils import set_seed

def load_models() -> None:
    """
    Load the model and tokenizer
    """

    # manual seed for reproducibility
    set_seed(42)

    # check device availability (save resources)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:

        device = "cpu"

    print(f"Using device: {device}")

    # model name
    model_name = "Qwen/Qwen3-0.6B"

    # # Check if flash attention is available
    use_flash_attention = False
    try:
        import flash_attn
        print("Flash Attention 2 is available and will be used")
        use_flash_attention = True
    except ImportError:
        print("Flash Attention 2 is not available, using standard attention")

    # Add flash attention to config if available
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device,
        "output_hidden_states":True
    }

    if use_flash_attention:
        # Flash Attention requires either float16 or bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            # Use bfloat16 for Ampere or newer GPUs (compute capability 8.0+)
            model_kwargs["torch_dtype"] = torch.bfloat16
            print("Using bfloat16 precision with Flash Attention")
        else:
            # Use float16 for older GPUs
            model_kwargs["torch_dtype"] = torch.float16
            print("Using float16 precision with Flash Attention")

        model_kwargs["attn_implementation"] = "flash_attention_2"

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) # set padding side if batching

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and tokenizer loaded.")