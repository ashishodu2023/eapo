# eapo/models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def _resolve_bnb_compute_dtype(choice: str) -> torch.dtype:
    choice = (choice or "auto").lower()
    if choice == "auto":
        return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if choice == "fp32":
        return torch.float32
    return torch.float16

def _explicit_device_map() -> dict:
    """Return an explicit module-wise device map to avoid accelerate calling model.to(...)."""
    if torch.cuda.is_available():
        return {"": "cuda:0"}  # place whole model on GPU 0
    else:
        return {"": "cpu"}

def load_model(
    model_id: str,
    device_map: str = "auto",            # ignored for quantized path; we use explicit dict
    quantization: str = "4bit",
    trust_remote_code: bool = True,
    attn_implementation: str = "eager",  # Phi-3 requires "eager" unless flash-attn is installed
    bnb_compute_dtype: str = "auto",
):
    """
    IMPORTANT:
    - For 4/8-bit, pass an EXPLICIT device_map dict (e.g., {"": "cuda:0"}) so accelerate does NOT call model.to(...).
    - Do NOT call model.to(...) after loading a quantized model.
    """

    # ----- Tokenizer -----
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ----- Build kwargs safely -----
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,  # "eager" or "flash_attention_2"
        "low_cpu_mem_usage": True,
    }

    if quantization in ("4bit", "8bit"):
        # EXPLICIT device map to avoid .to(...)
        model_kwargs["device_map"] = _explicit_device_map()

        if quantization == "4bit":
            # Recommended: use BitsAndBytesConfig (new API); but this path works across versions.
            # If you want to silence the deprecation warning, switch to quantization_config below.
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": _resolve_bnb_compute_dtype(bnb_compute_dtype),
            })
        else:
            model_kwargs.update({"load_in_8bit": True})

        # Do NOT pass torch_dtype for quantized path.
    else:
        # Full precision path
        base_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        model_kwargs.update({
            "torch_dtype": base_dtype,
            # For non-quantized, we can use a string or an explicit map; keep it simple:
            "device_map": _explicit_device_map(),
        })

    # ----- Load model -----
    print("[EAPO] from_pretrained kwargs:", {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                                            for k, v in model_kwargs.items()})
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # Prefer fast matmul if available (no-op if unsupported)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return tok, model
