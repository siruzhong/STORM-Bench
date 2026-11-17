from __future__ import annotations

import json
import hashlib
import pytest
from pathlib import Path

from scripts.eval.eval_revisitqa_model import extract_choice, grouped_chunk, iter_video_groups
from scripts.eval.data_io import dataset_manifest
from scripts.eval.revisitqa_streaming import pad_images_to_multiple
from scripts.eval.revisitqa_model_registry import MODEL_SPECS, parse_model_keys
from scripts.eval.run_revisitqa_models import RUNTIME_PYTHONS, merge_results
from scripts.download_revisitqa_models import is_complete_model_dir
from qwen_vl_utils import process_vision_info


def test_registry_contains_the_fifteen_model_suite():
    assert list(MODEL_SPECS) == [
        "qwen25_vl_3b",
        "qwen25_vl_7b",
        "qwen3_vl_4b",
        "qwen3_vl_8b",
        "qwen3_vl_32b",
        "qwen35_4b",
        "qwen35_9b",
        "internvl35_8b",
        "molmo2_8b",
        "minicpm_v45",
        "internvideo25_8b",
        "eagle25_8b",
        "videollama3_7b",
        "llava_next_video_7b",
        "glm41v_9b",
    ]
    assert {spec.runtime for spec in MODEL_SPECS.values()} == {
        "vlm",
        "videollama3",
        "modern",
        "legacy",
    }


def test_dataset_manifest_records_file_and_id_fingerprints(tmp_path):
    path = tmp_path / "questions.jsonl"
    rows = [
        {"id": "q1", "episode_id": "ep1", "video_path": "ep1.mp4"},
        {"id": "q2", "episode_id": "ep1", "video_path": "ep1.mp4"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    metadata = dataset_manifest(path)

    assert metadata["dataset_count"] == 2
    assert metadata["dataset_episode_count"] == 1
    assert metadata["dataset_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["dataset_ids_sha256"] == hashlib.sha256(b"q1\nq2").hexdigest()


def test_clock_video_metadata_matches_huggingface_video_metadata_fields():
    from scripts.eval.revisitqa_model_adapters import _clock_video_metadata

    times = [0.0, 1.0, 2.0, 2.0]
    metadata = _clock_video_metadata(len(times), 1.0, times)

    assert "timestamps" not in metadata
    assert metadata["fps"] == 1.0
    assert metadata["frames_indices"] == [0, 1, 2, 2]
    assert metadata["video_backend"] == "frames"
    reconstructed = [index / metadata["fps"] for index in metadata["frames_indices"]]
    assert reconstructed == times

    try:
        from transformers.video_utils import VideoMetadata
    except ImportError:
        return
    video_metadata = VideoMetadata(**metadata)
    assert list(video_metadata.timestamps) == times


def test_minicpm_adapter_forces_deterministic_decoding():
    source = Path("scripts/eval/revisitqa_model_adapters.py").read_text(encoding="utf-8")
    assert "max_new_tokens=8,\n                sampling=False," in source


def test_frame_alignment_pads_to_internvideo_temporal_groups():
    from PIL import Image

    images = [Image.new("RGB", (8, 8)) for _ in range(17)]
    timestamps = [float(index) for index in range(17)]
    padded, times = pad_images_to_multiple(images, timestamps, 4)
    assert len(padded) == 20
    assert len(times) == 20
    two, two_times = pad_images_to_multiple(images[:2], timestamps[:2], 4)
    assert len(two) == 4
    identity, identity_times = pad_images_to_multiple(images, timestamps, 1)
    assert len(identity) == 17
    assert identity_times == timestamps


def test_registry_parser_rejects_unknown_models():
    assert parse_model_keys("qwen3_vl_4b,molmo2_8b") == ["qwen3_vl_4b", "molmo2_8b"]
    try:
        parse_model_keys("missing")
    except ValueError as exc:
        assert "Unknown model keys" in str(exc)
    else:
        raise AssertionError("unknown model key was accepted")


def test_choice_extraction_accepts_common_model_outputs():
    assert extract_choice("A") == "A"
    assert extract_choice("答案：C") == "C"
    assert extract_choice("The answer is (D).") == "D"
    assert extract_choice("<think>Option A looks possible.</think><answer>C</answer>") == "C"
    assert extract_choice("无法判断") is None


def test_grouped_chunks_keep_video_questions_together():
    rows = [
        {"id": "a2", "video_path": "a.mp4", "query_time": 2},
        {"id": "b1", "video_path": "b.mp4", "query_time": 1},
        {"id": "a1", "video_path": "a.mp4", "query_time": 1},
    ]

    assert [row["id"] for row in grouped_chunk(rows, 2, 0)] == ["a1", "a2"]
    assert [row["id"] for row in grouped_chunk(rows, 2, 1)] == ["b1"]
    assert grouped_chunk(rows, 4, 2) == []
    assert grouped_chunk(rows, 4, 3) == []
    assert [[row["id"] for row in group] for group in iter_video_groups(grouped_chunk(rows, 1, 0))] == [
        ["a1", "a2"],
        ["b1"],
    ]


ABSTAIN = "Cannot be determined from the video up to this point."


def _v2_gt(sample_id, status, answer_index, intensity=2):
    return {
        "id": sample_id,
        "episode_id": "RevisitQA_cook_01",
        "query_time": 5.0,
        "evidence_spans": [[1.0, 2.0]],
        "question": f"question {sample_id}?",
        "options": ["Red", "Blue", "Green", ABSTAIN],
        "answer_index": answer_index,
        "abstention_index": 3,
        "change_intensity": intensity,
        "question_type": "current_state",
        "diagnostics": {
            "epistemic_status": status,
            "uncertainty_sources": [] if status == "known" else ["missing_observation"],
        },
    }


def test_merge_results_without_ground_truth_only_counts(tmp_path):
    rows = [
        {"id": "1", "pred_index": 0, "pred_choice": "A", "correct": True, "question_type": "state"},
        {"id": "2", "pred_index": None, "pred_choice": None, "correct": False, "question_type": "state"},
    ]
    (tmp_path / "pred_1_0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = merge_results(tmp_path, 1)

    assert summary["count"] == 2
    assert "revisitqa_score" not in summary
    assert (tmp_path / "pred.jsonl").exists()


def test_merge_results_scores_selective_qa(tmp_path):
    from scripts.eval.data_io import normalize_mcq_sample

    gt_path = tmp_path / "questions.jsonl"
    gt_rows = [_v2_gt("RevisitQA_cook_01_q01", "known", 0), _v2_gt("RevisitQA_cook_01_q02", "uncertain", 3, intensity=8)]
    gt_path.write_text("".join(json.dumps(row) + "\n" for row in gt_rows), encoding="utf-8")
    normalized = [normalize_mcq_sample(row) for row in gt_rows]
    signature = json.dumps({"protocol": "online"})
    for index, (gt, choice) in enumerate(zip(normalized, (normalized[0]["answer"], 3))):
        (tmp_path / f"pred_2_{index}.jsonl").write_text(
            json.dumps({
                "id": gt["id"],
                "episode_id": gt["episode_id"],
                "pred_index": choice,
                "pred_choice": "ABCD"[choice],
                "eval_signature": signature,
                "stream_frames_seen": 8 + index,
                "buffered_frames": 8 + index,
                "buffer_capacity": None,
                "future_leakage": False,
                "visual_state_persistent": True,
            }) + "\n",
            encoding="utf-8",
        )

    summary = merge_results(tmp_path, 2, gt_file=gt_path, bootstrap_replicates=0)
    written = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert summary["effective_reliability"] == written["effective_reliability"] == 0.5
    assert summary["known_accuracy"] == 1.0
    assert summary["abstention_recall"] == 1.0
    assert "revisitqa_score" not in summary
    assert summary["streaming_audit"]["max_stream_frames_seen"] == 9


def test_download_completeness_requires_every_indexed_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "model-1.safetensors").write_bytes(b"one")
    assert not is_complete_model_dir(tmp_path)

    (tmp_path / "model-2.safetensors").write_bytes(b"two")
    assert is_complete_model_dir(tmp_path)


def test_vendored_qwen_vision_info_preserves_legacy_contract():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    assert process_vision_info(messages) == (None, None)


def test_revisitqa_runner_keeps_model_runtimes_isolated():
    assert RUNTIME_PYTHONS["vlm"].parents[1].name == ".venv-revisitqa-vlm"
    assert RUNTIME_PYTHONS["videollama3"].parents[1].name == ".venv-revisitqa-videollama3"
    assert RUNTIME_PYTHONS["modern"].parents[1].name == ".venv-revisitqa-modern"
    assert RUNTIME_PYTHONS["legacy"].parents[1].name == ".venv-revisitqa-legacy"


def test_revisitqa_runners_register_seven_released_splits():
    from scripts.eval.revisitqa_datasets import REVISITQA_SUBSETS, revisitqa_paths
    from scripts.eval.run_eval_offline import DATASETS
    from scripts.eval.run_eval_online import PUBLIC_DATASETS

    assert set(REVISITQA_SUBSETS) == {"cook", "bike", "healthy", "music", "sports", "thor", "vhome"}
    for subset in REVISITQA_SUBSETS:
        gt, video = revisitqa_paths(subset)
        key = f"revisitqa_{subset}"
        assert DATASETS[key]["qa"] == PUBLIC_DATASETS[key]["gt_file"] == gt
        assert DATASETS[key]["root"] == PUBLIC_DATASETS[key]["video_dir"] == video


@pytest.mark.parametrize("gt_status", ["known", "uncertain"])
def test_text_only_eval_skips_video_and_uses_answer_text(tmp_path, monkeypatch, gt_status):
    from argparse import Namespace

    from scripts.eval import eval_revisitqa_model as mod

    ckpt = tmp_path / "ckpt" / "Qwen3.5-9B"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    gt = tmp_path / "questions.jsonl"
    gt.write_text(
        json.dumps(
            {
                "id": "q1",
                "episode_id": "RevisitQA_vhome_01",
                "query_time": 1.0,
                "question": "What color is the tool?",
                "options": ["Red", "Blue", "Green", ABSTAIN],
                "answer_index": 0 if gt_status == "known" else 3,
                "abstention_index": 3,
                "evidence_spans": [[0.0, 1.0]],
                "change_intensity": 2,
                "diagnostics": {"epistemic_status": gt_status, "uncertainty_sources": [] if gt_status == "known" else ["missing_observation"]},
                "question_type": "factual_retrieval",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeAdapter:
        def answer_text(self, question: str) -> str:
            assert "What color is the tool?" in question
            return "A"

        def answer(self, *args, **kwargs):
            raise AssertionError("video answer path should not run in text-only eval")

    monkeypatch.setattr(mod, "load_adapter", lambda *args, **kwargs: FakeAdapter())
    monkeypatch.setattr(
        mod,
        "resolve_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("video should not be opened")),
    )

    args = Namespace(
        model_key="qwen35_9b",
        checkpoint_root=str(tmp_path / "ckpt"),
        protocol="offline",
        text_only=True,
        gt_file=str(gt),
        video_root=str(tmp_path / "missing_videos"),
        output_dir=str(tmp_path / "out"),
        num_chunks=1,
        chunk_idx=0,
        fps=1.0,
        max_pixels=200704,
        buffer_frames=0,
        sample_limit=None,
        episode_limit=None,
        seed=20260817,
        glm_thinking_budget=512,
        bootstrap_replicates=0,
        overwrite=True,
    )
    mod.run_revisitqa_protocol(args)
    pred = json.loads((tmp_path / "out" / "pred.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert pred["pred_choice"] == "A"
    assert pred["buffered_frames"] == 0
    assert pred["buffer_timestamps"] == []
    assert pred["protocol"] == "offline_text_only"
    assert pred["future_leakage"] is False
    assert '"text_only": true' in pred["eval_signature"]


def test_videollama3_adapter_requires_official_flash_attention_2():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/eval/revisitqa_model_adapters.py"
    ).read_text(encoding="utf-8")
    assert '_require_flash_attention_2("VideoLLaMA3-7B")' in source
    assert "_assert_videollama3_flash_vision(self.model)" in source
    assert 'attn_implementation=attn' in source
    assert "VisionFlashAttention2" in source
    # Must not silently fall back to the dense-mask SDPA path that OOMs.
    video_block = source.split("class VideoLLaMA3Adapter:", 1)[1].split("class ", 1)[0]
    assert "sdpa" not in video_block
