"""vLLM V1 引擎级自定义 LogitsProcessor（vllm>=0.10.2）。只在云端 import（依赖 vllm）。核心逻辑在 rollout.SpanBatchState。
注册：LLM(..., logits_processors=[SpanLogitsProcessor])；每请求配置：SamplingParams(extra_args={"cot_span": {"cap","reserve","mode"}})。
V1 引擎核心可能在子进程里 → 本模块必须可 import：PYTHONPATH 含项目根、COT_NO_UNSLOTH=1（setup/env_cloud.sh）。"""
from __future__ import annotations
from typing import Optional
import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality
from .rollout import SpanBatchState

class SpanLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        from transformers import AutoTokenizer
        mc = vllm_config.model_config
        tok = AutoTokenizer.from_pretrained(mc.tokenizer)
        self.state = SpanBatchState(tok, int(mc.get_vocab_size()), device=device)

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            return
        reqs = self.state.reqs
        for idx in batch_update.removed:
            reqs.pop(idx, None)
        for idx, params, _prompt_ids, out_ids in batch_update.added:
            e = self.state.make_entry(getattr(params, "extra_args", None), out_ids)
            if e is None:
                reqs.pop(idx, None)
            else:
                reqs[idx] = e
        for a, b, direction in batch_update.moved:
            ea, eb = reqs.pop(a, None), reqs.pop(b, None)
            if ea is not None:
                reqs[b] = ea
            if direction == MoveDirectionality.SWAP and eb is not None:
                reqs[a] = eb

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return self.state.apply(logits) if self.state.reqs else logits
