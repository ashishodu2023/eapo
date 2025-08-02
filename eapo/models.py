import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def _resolve_bnb_compute_dtype(choice: str) -> torch.dtype | None:
    """
    Maps config string to torch dtype. 'auto' => bf16 if supported else fp16.
    """
    choice = (choice or "auto").lower()
    if choice == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if choice == "fp32":
        return torch.float32
    return torch.float16

def load_model(
    model_id: str,
    device_map: str = "auto",
    quantization: str = "4bit",
    trust_remote_code: bool = True,
    attn_implementation: str = "eager",   # <- use "eager" for Phi-3; "sdpa"/"flash_attention_2" if supported
    bnb_compute_dtype: str = "auto",
):
    """
    Load tokenizer + model with optional 4/8-bit quantization and chosen attention kernel.

    IMPORTANT:
    - Do NOT call model.to(...) after loading in 4/8-bit; it's unsupported by bitsandbytes.
    - Move INPUTS to the model device instead (see evaluator.py).
    """
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Quantization config
    quant_cfg = None
    if quantization == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_resolve_bnb_compute_dtype(bnb_compute_dtype),
        )
    elif quantization == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    # Base dtype (only used when NOT quantized)
    base_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Build model — let accelerate place shards via device_map; DO NOT call .to(...)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,                   # "auto" recommended
        quantization_config=quant_cfg,           # None for full-precision
        torch_dtype=base_dtype if quant_cfg is None else None,
        attn_implementation=attn_implementation, # "eager" for Phi-3; "sdpa"/"flash_attention_2" if supported
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # Prefer fast matmul kernels when available (safe no-ops otherwise)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return tok, model
