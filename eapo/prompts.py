from dataclasses import dataclass

@dataclass
class PromptConfig:
    style: str          # "concise" | "role" | "stepwise"
    reasoning: str      # "none" | "brief" | "bounded"
    format: str         # "free" | "bullets" | "json"
    brevity: str        # "none" | "1sent" | "3sent" | "word50"

def render_prompt(cfg: PromptConfig, doc: str) -> str:
    sys = ""
    if cfg.style == "concise":
        sys = "You are a precise assistant. Answer succinctly."
    elif cfg.style == "role":
        sys = "You are an expert editor. Provide accurate, concise responses."
    elif cfg.style == "stepwise":
        sys = "Follow steps carefully and avoid unnecessary words."

    reason = ""
    if cfg.reasoning == "brief":
        reason = "Think briefly and only if needed."
    elif cfg.reasoning == "bounded":
        reason = "If reasoning is needed, limit it to two short steps."

    outfmt = ""
    if cfg.format == "bullets":
        outfmt = "Return 3 bullet points."
    elif cfg.format == "json":
        outfmt = 'Return a JSON with keys {"summary": "..."} .'

    brev = ""
    if cfg.brevity == "1sent":
        brev = "Use at most one sentence."
    elif cfg.brevity == "3sent":
        brev = "Use at most three sentences."
    elif cfg.brevity == "word50":
        brev = "Keep it under 50 words."

    instruction = f"{sys} {reason} {outfmt} {brev}".strip()
    return f"""[SYSTEM]
{instruction}

[USER]
Summarize the following passage:
\"\"\"{doc}\"\"\"
"""

STYLE = ["concise", "role", "stepwise"]
REASON = ["none", "brief", "bounded"]
FORMAT = ["free", "bullets", "json"]
BREVITY = ["none", "1sent", "3sent", "word50"]
