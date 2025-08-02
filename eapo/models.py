import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_id: str, device_map: str = "auto", quantization: str = "4bit", trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = None
    if quantization == "4bit":
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code
    )
    model.eval()
    return tok, model
