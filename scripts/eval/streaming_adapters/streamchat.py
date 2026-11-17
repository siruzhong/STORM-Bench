"""StreamChat (ICLR 2025) on LongVA-7B with its online hierarchical memory.

Upstream StreamChat encodes the incoming frames once, keeps a short-term memory sampled from the most recent frames,
captions every chunk of 40 frames into a memory-tree leaf (summarized into a parent node every 10 leaves), and
answers by retrieving the tree path most similar to the question (ColBERT text embeddings) together with the
short-term memory. Here `observe` encodes each newly visible 1 FPS frame once and turns every completed 40-frame
chunk into a leaf once; `answer` captions the still incomplete trailing chunk as a temporary leaf (cached by its
length, discarded when the chunk completes), then uses the wrapper's retrieval, prompt and decoding.

Model loading (two resident replicas: answerer on the first visible GPU, captioner on the second), budget and
prompts are those of `StreamChatAdapter` in `scripts/eval/public_baseline_adapters.py` (upstream f8bcd22). The
upstream frame sampler (0.2 x source fps) is replaced by the benchmark clock (1 FPS), so a 40-frame chunk spans 40 s.
"""

from __future__ import annotations

from scripts.eval.streaming_adapters import StreamingSpec

SPEC = StreamingSpec(
    key="streamchat_7b",
    checkpoint="LongVA-7B-DPO",
    runtime=".venv-streamchat/bin/python",
    description="StreamChat on LongVA-7B, memory tree updated per 40-frame chunk (hmxiong/StreamChat f8bcd22); needs 2 GPUs",
)


class StreamChatStreamingAdapter:
    def __init__(self, model_path):
        import torch

        from scripts.eval.public_baseline_adapters import PUBLIC_BASELINES, StreamChatAdapter

        if torch.cuda.device_count() < 2:
            raise RuntimeError("StreamChat needs two visible GPUs; pass a device group such as --cuda-devices 0+1")
        self.torch = torch
        self.inner = StreamChatAdapter(PUBLIC_BASELINES[SPEC.key], model_path)
        self.chunk = int(self.inner.budget["chunk_size"])
        self.reset()

    def reset(self) -> None:
        self.features = []
        self.leaves = []
        self.partial = None  # (frame count, leaf) for the incomplete trailing chunk

    def _leaf(self, chunk):
        from scripts.eval.public_baseline_adapters import _StreamChatNode

        return _StreamChatNode(self.torch.cat(chunk, dim=0), self.inner._caption_chunk(chunk))

    def observe(self, frames) -> None:
        if not frames:
            return
        self.features += self.inner._encode_frames([f.image.convert("RGB") for f in frames])
        while len(self.features) >= (len(self.leaves) + 1) * self.chunk:
            start = len(self.leaves) * self.chunk
            self.leaves.append(self._leaf(self.features[start:start + self.chunk]))
            self.partial = None

    def _tree(self):
        nodes = list(self.leaves)
        rest = self.features[len(self.leaves) * self.chunk:]
        if rest:
            if self.partial is None or self.partial[0] != len(rest):
                self.partial = (len(rest), self._leaf(rest))
            nodes.append(self.partial[1])
        interval = int(self.inner.budget["tree_interval"])
        if len(nodes) >= interval:  # upstream: summarize the first `interval` leaves into one parent node
            raise NotImplementedError("episodes longer than tree_interval chunks are not expected at 1 FPS")
        return nodes

    def answer(self, prompt: str, option_count: int) -> str:
        if not self.features:
            raise RuntimeError("StreamChat memory holds no frames before answering")
        inner = self.inner
        inner._tree = self._tree()
        inner._index_tree_text(inner._tree)
        inner._short_memory = inner._select_short_memory(self.features)
        inner._cached_media = _NO_MEDIA  # skip the wrapper's offline media loading
        return inner.answer(_NO_MEDIA, prompt, option_count)

    def state_info(self) -> dict:
        nbytes = sum(t.numel() * t.element_size() for t in self.features)
        return {"state_frames": len(self.features), "state_bytes": int(nbytes)}


class _NoMedia:
    def resolve(self):
        return self


_NO_MEDIA = _NoMedia()


def build(model_path, args) -> StreamChatStreamingAdapter:
    return StreamChatStreamingAdapter(model_path)
