from argparse import Namespace
from pathlib import Path

from PIL import Image

from scripts.eval.eval_revisitqa_model import build_eval_signature, protocol_label
from scripts.eval.revisitqa_streaming import (
    PROTOCOL_NAME,
    SlidingWindow,
    StreamFrame,
    diagnostic_cell,
    frames_to_timestamps,
    keep_last_frames,
    pad_images_to_multiple,
    select_episode_subset,
)


def _frame(index: int) -> StreamFrame:
    return StreamFrame(index, float(index), Image.new("RGB", (16, 16), (index, 0, 0)))


def test_sliding_window_keeps_the_most_recent_frames():
    window = SlidingWindow(3)
    window.update([_frame(index) for index in range(3)])
    window.update([_frame(index) for index in range(3, 10)])

    assert [frame.stream_index for frame in window.snapshot()] == [7, 8, 9]
    assert [frame.timestamp for frame in window.snapshot()] == [7.0, 8.0, 9.0]
    assert window.seen == 10
    assert window.num_evictions == 7
    assert window.limit == 3


def test_unbounded_window_keeps_the_whole_clock_stream():
    window = SlidingWindow(0)
    window.update([_frame(index) for index in range(10)])

    assert [frame.stream_index for frame in window.snapshot()] == list(range(10))
    assert window.limit is None
    assert window.seen == 10
    assert window.num_evictions == 0


def test_protocol_helpers_pad_instead_of_redrawing_times():
    images = [_frame(index).image for index in range(17)]
    timestamps = [float(index) for index in range(17)]

    padded_images, padded_times = pad_images_to_multiple(images, timestamps, 4)
    assert len(padded_images) == 20
    assert padded_times[:17] == timestamps
    assert padded_times[17:] == [timestamps[-1]] * 3
    assert padded_images[-3:] == [images[-1]] * 3

    kept_images, kept_times, capped = keep_last_frames(images, timestamps, 8)
    assert capped is True
    assert [image.getpixel((0, 0))[0] for image in kept_images] == list(range(9, 17))
    assert kept_times == timestamps[-8:]
    assert frames_to_timestamps([_frame(2), _frame(5)]) == [2.0, 5.0]


def test_episode_subset_preserves_all_diagnostic_cells_and_query_updates():
    rows = []
    cells = [(2, "known"), (1, "uncertain"), (5, "known"), (9, "known")]
    for index, (intensity, status) in enumerate(cells):
        rows.append({
            "id": f"best-{index}",
            "episode_id": "best",
            "query_time": float(index + 1),
            "change_intensity": intensity,
            "diagnostics": {"epistemic_status": status},
        })
    rows.append({
        "id": "other", "episode_id": "other", "query_time": 1.0,
        "change_intensity": 2,
        "diagnostics": {"epistemic_status": "known"},
    })

    selected = select_episode_subset(rows, 1, 7)
    assert {row["episode_id"] for row in selected} == {"best"}
    assert {diagnostic_cell(row) for row in selected} == {
        ("1-3", "known"), ("1-3", "uncertain"), ("4-6", "known"), ("7-10", "known")
    }


def test_protocol_labels():
    assert protocol_label("offline", 0) == "offline"
    assert protocol_label("online", 0) == "online_all"
    assert protocol_label("online", 16) == "online_b16"
    assert protocol_label("offline", 0, True) == "offline_text_only"
    assert protocol_label("online", 0, True) == "online_all_text_only"
    assert protocol_label("online", 16, True) == "online_all_text_only"


def test_text_only_signature_declares_text_input_and_no_visual_buffer(tmp_path):
    args = Namespace(
        protocol="offline",
        text_only=True,
        buffer_frames=0,
        fps=1.0,
        max_pixels=200704,
        model_key="qwen35_9b",
        episode_limit=None,
    )
    data, encoded = build_eval_signature(args, tmp_path)
    assert data["protocol"] == "offline"
    assert data["protocol_label"] == "offline_text_only"
    assert data["model_input"] == "text"
    assert data["text_only"] is True
    assert data["buffer_policy"] is None
    assert data["buffer_frames"] is None
    assert '"text_only": true' in encoded


def test_eval_signature_declares_shared_clock_and_frame_list_input(tmp_path):
    args = Namespace(
        protocol="online",
        buffer_frames=0,
        fps=1.0,
        max_pixels=200704,
        model_key="qwen25_vl_3b",
        episode_limit=None,
    )
    data, encoded = build_eval_signature(args, tmp_path)
    assert data["schema_version"] == PROTOCOL_NAME
    assert data["protocol_label"] == "online_all"
    assert data["buffer_policy"] == "sliding_window"
    assert data["buffer_frames"] is None
    assert "max_observed_frames" not in data
    assert data["model_input"] == "frame_list"
    assert data["temporal_resample"] is False
    assert data["native_recurrent_state"] is False
    assert data["question_history_reused"] is False
    assert '"temporal_resample": false' in encoded


def test_eval_signature_carries_manifest_identity(tmp_path):
    args = Namespace(
        protocol="offline",
        buffer_frames=0,
        fps=1.0,
        max_pixels=200704,
        model_key="qwen25_vl_3b",
        episode_limit=None,
        sample_limit=None,
    )
    data, _ = build_eval_signature(
        args,
        tmp_path,
        "file-hash",
        {
            "dataset_sha256": "file-hash",
            "dataset_ids_sha256": "id-hash",
            "dataset_count": 2,
            "dataset_episode_count": 1,
        },
    )
    assert data["dataset_sha256"] == "file-hash"
    assert data["dataset_ids_sha256"] == "id-hash"
    assert data["dataset_count"] == 2
    assert data["dataset_episode_count"] == 1


def test_runners_declare_external_state_without_native_recurrence_or_old_protocols():
    unified = Path("scripts/eval/eval_revisitqa_model.py").read_text(encoding="utf-8")
    streaming = Path("scripts/eval/run_revisitqa_streaming_models.py").read_text(encoding="utf-8")
    runner = Path("scripts/eval/run_revisitqa_models.py").read_text(encoding="utf-8")
    clock = Path("scripts/eval/revisitqa_streaming.py").read_text(encoding="utf-8")
    adapters = Path("scripts/eval/revisitqa_model_adapters.py").read_text(encoding="utf-8")
    reliability = Path("scripts/eval/revisitqa_reliability.py").read_text(encoding="utf-8")

    assert 'PROTOCOL_NAME = "clock_1fps_sliding_window"' in clock
    assert "class SlidingWindow" in clock
    assert "DeterministicReservoir" not in clock
    assert "materialize_buffer_video" not in clock
    assert "native_recurrent_state" in unified
    assert "question_history_reused" in unified
    assert "adapter.THINKING_BUDGET = args.glm_thinking_budget" in unified
    assert "--buffer-frames" in streaming
    assert '"--buffer-frames", "16"' in streaming
    assert "--max-frames" not in unified
    assert "--max-frames" not in streaming
    assert "--max-frames" not in runner
    assert "score_selective" in runner
    assert "score_selective" in reliability
    assert "revisitqa_score" not in runner
    assert "cell_score" not in reliability
    assert "--with-status" not in unified
    assert "--with-status" not in runner
    assert "--video-root" in runner
    assert "from scripts.eval.run_revisitqa_models import main" in streaming
    assert "max_observed_frames" not in unified
    assert "max_observed_frames" not in clock
    assert "causal_clip" not in unified
    assert "causal_prefix_reencode" not in unified
    assert "do_sample_frames" in adapters
    assert "video_path" not in adapters
    assert "linspace" not in adapters
    assert not Path("scripts/eval/revisitqa_diagnostics.py").exists()
    assert not Path("scripts/eval/eval_revisitqa_diagnostics.py").exists()
