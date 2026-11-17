"""ReKV (ICLR 2025) on LLaVA-OneVision-7B with its streaming KV cache.

Upstream ReKV encodes the video stream chunk by chunk into a KV cache (recent tokens on GPU, older blocks offloaded)
and, at question time, retrieves the top-k frame blocks by the question's attention and answers from them; the
question pass runs in retrieval mode, which does not write its KV into the cache. Here `observe` encodes each newly
visible frame once (`encode_video`), `answer` calls the upstream `question_answering`, and the cache is cleared and
the system prompt re-encoded only at an episode boundary.

Loading, budget (n_local 15000, top-k 64, chunk size 1) and the multiple-choice prompt are the ones of the offline
wrapper `ReKvAdapter` in `scripts/eval/public_baseline_adapters.py` (upstream 1fd9a3d). Deviation: frames are
encoded at the benchmark clock (1 FPS) instead of the 0.5 FPS upstream uses for long videos, so every streaming
model receives the same frames, including the one at the query time.
"""

from __future__ import annotations

from scripts.eval.streaming_adapters import StreamingSpec

SPEC = StreamingSpec(
    key="rekv_llava_ov_7b",
    checkpoint="llava-onevision-qwen2-7b-ov-hf",
    runtime=".venv-revisitqa-videollama3/bin/python",
    description="ReKV on LLaVA-OneVision-7B, KV cache updated one frame at a time (Becomebright/ReKV 1fd9a3d)",
)


class ReKVStreamingAdapter:
    def __init__(self, model_path):
        import numpy as np
        import torch

        from scripts.eval.public_baseline_adapters import PUBLIC_BASELINES, ReKvAdapter

        self.np, self.torch = np, torch
        self.inner = ReKvAdapter(PUBLIC_BASELINES[SPEC.key], model_path)
        self.model = self.inner.model
        self.reset()

    def reset(self) -> None:
        self.model.clear_cache()
        self.model.encode_init_prompt()
        self.frame_count = 0

    def observe(self, frames) -> None:
        if not frames:
            return
        video = self.np.stack([self.np.asarray(f.image.convert("RGB")) for f in frames])
        self.model.encode_video(self.torch.from_numpy(video))
        self.frame_count += len(frames)

    def answer(self, prompt: str, option_count: int) -> str:
        if self.frame_count == 0:
            raise RuntimeError("ReKV cache holds no frames before answering")
        labels = ", ".join(chr(65 + i) for i in range(option_count))
        question = f"Question: {prompt}\nOnly give the best option ({labels})."
        text = {"question": prompt, "prompt": self.model.get_prompt(question, mc=True)}
        return self.model.question_answering(text, max_new_tokens=16).strip()

    def state_info(self) -> dict:
        return {"state_frames": self.frame_count, "state_bytes": int(self.model.calc_memory_usage())}


def build(model_path, args) -> ReKVStreamingAdapter:
    return ReKVStreamingAdapter(model_path)
