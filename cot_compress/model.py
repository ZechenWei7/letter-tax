"""模型加载（Unsloth）。"""
from __future__ import annotations
import cot_compress  # noqa: F401  unsloth first
import torch

def load_model(cfg: dict, for_inference: bool = True):
    """model.use_unsloth（默认 true）：Unsloth FastLanguageModel；false：transformers bf16（云端 TRL 原生路径）。"""
    m = cfg["model"]
    if not m.get("use_unsloth", True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m["name"])
        model = AutoModelForCausalLM.from_pretrained(m["name"], dtype=torch.bfloat16, device_map="cuda")
        if for_inference:
            model.eval()
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return model, tok
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=m["name"],
        max_seq_length=int(m.get("max_seq_length", 4096)),
        load_in_4bit=bool(m.get("load_in_4bit", True)),
        fast_inference=bool(m.get("fast_inference", False)),
        dtype=m.get("dtype"),
    )
    if for_inference:
        FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok
