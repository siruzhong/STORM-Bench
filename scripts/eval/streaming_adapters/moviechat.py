"""MovieChat (CVPR 2024) in its breakpoint (streaming) mode.

Upstream MovieChat reads the video fragment by fragment: each fragment's frame tokens enter the short-term memory,
which is merged by adjacent-frame similarity and flushed into the long-term memory (`encode_short_memory_frame`).
In breakpoint mode (`middle_video=True`) a question at time t is answered from long-term memory, the last
short-term snapshot and the current frame (`encode_long_video(cur_image, True)`).

Here `observe` collects the newly visible 1 FPS frames and pushes every complete 8-frame fragment (upstream
`max_frames_per_fragment`) into memory once. `answer` encodes the still incomplete fragment and the frame at t_q the
way the breakpoint path does, on a copy of the memory, then restores the memory, because upstream
`encode_long_video` and a partial fragment would otherwise change the state used by later questions; frames of an
incomplete fragment are therefore encoded again once the fragment completes. Loading, prompt and decoding (sampling
with seed 42, as upstream) are those of the offline wrapper `MovieChatAdapter` in
`scripts/eval/public_baseline_adapters.py` (upstream ba9bb80).
"""

from __future__ import annotations

from scripts.eval.streaming_adapters import StreamingSpec

SPEC = StreamingSpec(
    key="moviechat_7b",
    checkpoint="MovieChat-vicuna",
    runtime=".venv-moviechat/bin/python",
    description="MovieChat-7B breakpoint mode, memory updated per 8-frame fragment (rese1f/MovieChat ba9bb80)",
)


class MovieChatStreamingAdapter:
    def __init__(self, model_path):
        import numpy as np
        import torch

        from scripts.eval.public_baseline_adapters import PUBLIC_BASELINES, MovieChatAdapter

        self.np, self.torch = np, torch
        self.inner = MovieChatAdapter(PUBLIC_BASELINES[SPEC.key], model_path)
        self.model = self.inner.model
        # A plain tensor attribute (not a buffer), so `.to("cuda")` leaves it on the CPU; upstream's offline branch
        # moves it per call, the breakpoint branch does not.
        self.model.frame_position_embeddings = self.model.frame_position_embeddings.to("cuda:0")
        self.fragment = int(self.inner.budget["max_frames_per_fragment"])
        self.reset()

    def reset(self) -> None:
        self.model.short_memory_buffer = []
        self.model.temp_short_memory = []
        self.model.long_memory_buffer = []
        self.pending = []
        self.last = None
        self.frame_count = 0

    def _tensor(self, images):
        """(1, 3, T, 224, 224), normalized as the wrapper's decord path."""
        from PIL import Image

        frames = self.np.stack([self.np.asarray(i.convert("RGB").resize((224, 224), Image.BICUBIC)) for i in images])
        video = self.torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        return ((video - self.inner.mean) / self.inner.std).unsqueeze(0).to("cuda:0")

    def observe(self, frames) -> None:
        for frame in frames:
            self.pending.append(frame.image)
            self.last = frame.image
            self.frame_count += 1
            if len(self.pending) == self.fragment:
                with self.torch.inference_mode():
                    self.model.encode_short_memory_frame(self._tensor(self.pending))
                self.pending = []

    def answer(self, prompt: str, option_count: int) -> str:
        if self.last is None:
            raise RuntimeError("MovieChat memory holds no frames before answering")
        model = self.model
        saved = (list(model.short_memory_buffer), list(model.temp_short_memory), list(model.long_memory_buffer))
        try:
            with self.torch.inference_mode():
                if self.pending:
                    model.encode_short_memory_frame(self._tensor(self.pending), len(self.pending))
                cur_image = model.encode_image(self._tensor([self.last]))
                self.inner._video_embedding, _ = model.encode_long_video(cur_image, True)
            # The wrapper's answer() only loads media when the path changes; pin the cache to skip loading.
            self.inner._cached_media = _NO_MEDIA
            return self.inner.answer(_NO_MEDIA, prompt, option_count)
        finally:
            model.short_memory_buffer, model.temp_short_memory, model.long_memory_buffer = saved

    def state_info(self) -> dict:
        tensors = self.model.long_memory_buffer + self.model.short_memory_buffer
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        return {"state_frames": self.frame_count, "state_bytes": int(nbytes)}


class _NoMedia:
    def resolve(self):
        return self


_NO_MEDIA = _NoMedia()


def build(model_path, args) -> MovieChatStreamingAdapter:
    return MovieChatStreamingAdapter(model_path)
