from PIL import Image

from scripts.eval.data_io import normalize_mcq_sample
from scripts.eval.revisitqa_reliability import score_selective
from scripts.eval.revisitqa_streaming import StreamFrame, blank_visible_frames

ABSTAIN = "Cannot be determined from the video up to this point."


def _row(sample_id, status, answer_index, intensity=2, spans=((1.0, 2.0),)):
    return {
        "id": sample_id,
        "episode_id": sample_id.split("_q")[0],
        "question": "Q?",
        "options": ["a", "b", "c", ABSTAIN],
        "answer_index": answer_index,
        "abstention_index": 3,
        "query_time": 20.0,
        "evidence_spans": [list(span) for span in spans],
        "change_intensity": intensity,
        "question_type": "current_state",
        "diagnostics": {"epistemic_status": status, "uncertainty_sources": [] if status == "known" else ["missing_observation"]},
    }


def test_normalization_keeps_abstention_last():
    for index in range(50):
        row = normalize_mcq_sample(_row(f"RevisitQA_cook_01_q{index:02d}", "known", index % 3))
        assert row["question"].splitlines()[-1] == f"(D) {ABSTAIN}"
        assert row["abstention_choice"] == 3
        assert row["answer"] != 3
        options = [line[4:] for line in row["question"].splitlines()[1:]]
        assert options[row["answer"]] == ["a", "b", "c"][index % 3]


def test_selective_scores():
    gt = [
        normalize_mcq_sample(_row("RevisitQA_cook_01_q01", "known", 0)),
        normalize_mcq_sample(_row("RevisitQA_cook_01_q02", "known", 1)),
        normalize_mcq_sample(_row("RevisitQA_cook_02_q01", "known", 2, spans=((0.0, 1.0),))),
        normalize_mcq_sample(_row("RevisitQA_cook_02_q02", "uncertain", 3, intensity=8)),
        normalize_mcq_sample(_row("RevisitQA_cook_02_q03", "uncertain", 3, intensity=8)),
    ]
    wrong = lambda g: (g["answer"] + 1) % 3
    preds = [
        {"id": gt[0]["id"], "pred_index": gt[0]["answer"], "pred_choice": "X"},  # correct answer: +1
        {"id": gt[1]["id"], "pred_index": wrong(gt[1]), "pred_choice": "X"},  # wrong answer: -1
        {"id": gt[2]["id"], "pred_index": 3, "pred_choice": "D"},  # false abstention: 0
        {"id": gt[3]["id"], "pred_index": 3, "pred_choice": "D"},  # correct abstention: 0
        {"id": gt[4]["id"], "pred_index": None, "pred_choice": None},  # invalid on Uncertain: -1
    ]
    score = score_selective(gt, preds, bootstrap_replicates=20)
    assert abs(score["effective_reliability"] - (-1 / 5)) < 1e-9
    assert abs(score["known_accuracy"] - 1 / 3) < 1e-9
    assert score["abstention_recall"] == 0.5
    assert abs(score["false_abstention_rate"] - 1 / 3) < 1e-9
    assert score["abstention_precision"] == 0.5
    assert score["coverage"] == 3 / 5
    assert abs(score["risk"] - 2 / 3) < 1e-9
    assert score["by_intensity"]["high"]["uncertain_count"] == 2
    assert score["by_evidence_age"]["ge10s"]["count"] == 5
    assert "revisitqa_score" not in score
    h = 2 * (1 / 3) * 0.5 / (1 / 3 + 0.5)
    assert abs(score["known_abstention_hmean"] - h) < 1e-9


def test_blank_frames_keep_shape_and_time():
    frames = [StreamFrame(i, float(i), Image.new("RGB", (8, 6), (200, 10, 10))) for i in range(3)]
    blank = blank_visible_frames(frames)
    assert [f.timestamp for f in blank] == [0.0, 1.0, 2.0]
    assert all(f.image.size == (8, 6) and f.image.getextrema() == ((0, 0), (0, 0), (0, 0)) for f in blank)


def test_options_only_prompt_drops_stem_and_preamble():
    from scripts.eval.eval_revisitqa_model import options_only_prompt

    question = "Use only video observations.\nQuestion: Where is the cup?\n(A) sink\n(B) table\n(C) shelf\n(D) " + ABSTAIN
    prompt = options_only_prompt(question)
    assert "Where is the cup" not in prompt and "Use only" not in prompt
    assert prompt.splitlines()[1:] == ["(A) sink", "(B) table", "(C) shelf", f"(D) {ABSTAIN}"]


def test_forced_choice_prompt_removes_abstention_line():
    from scripts.eval.eval_revisitqa_model import forced_choice_prompt

    question = "Where is the cup?\n(A) sink\n(B) table\n(C) shelf\n(D) " + ABSTAIN
    assert forced_choice_prompt(question, 3).splitlines() == ["Where is the cup?", "(A) sink", "(B) table", "(C) shelf"]
