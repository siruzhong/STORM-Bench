"""Keep retries bounded and prevent an accepted run from changing its program."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("rollout_retry", Path(__file__).parents[1] / "scripts/rollout_retry.py")
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


def arguments(root):
    config, plan = root / "source.yaml", root / "source.json"
    config.write_text("seed: 42\n")
    plan.write_text(json.dumps({"plan_id": "frozen"}))
    return SimpleNamespace(output=root / "runs", config=config, plan=plan,
                           executable=root / "simulator", max_attempts=3,
                           gpu_index=0, xorg_root=None, display=None)


def evidence(command, program):
    root = Path(command[command.index("--output") + 1])
    root.mkdir()
    for name, value in {"event_program.json": program,
                        "plan_execution.json": {"accepted": True},
                        "camera_continuity.json": {"accepted": True},
                        "observer_validation.json": {}, "debug_validation.json": {}}.items():
        (root / name).write_text(json.dumps(value))


def test_retries_freeze_inputs_and_stop_on_success(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = []

    def execute(command, **kwargs):
        plan = json.loads(Path(command[command.index("--plan") + 1]).read_text())
        assert plan == {"plan_id": "frozen"}
        assert Path(command[command.index("--config") + 1]).read_text() == "seed: 42\n"
        calls.append(command)
        args.plan.write_text('{"plan_id": "changed"}')
        args.config.write_text("seed: 7\n")
        if len(calls) == 1:
            return SimpleNamespace(returncode=1)
        evidence(command, plan)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(retry.subprocess, "run", execute)
    assert retry.run_attempts(args).name == "runs_attempt_02"
    report = json.loads((args.output / "attempts.json").read_text())
    assert len(calls) == 2
    assert [item["accepted"] for item in report["attempts"]] == [False, True]


@pytest.mark.parametrize("failure", ["exit_code", "changed_plan"])
def test_retry_limit_never_accepts_failed_or_changed_runs(tmp_path, monkeypatch, failure):
    args = arguments(tmp_path)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if failure == "changed_plan":
            evidence(command, {"plan_id": "different"})
        return SimpleNamespace(returncode=1 if failure == "exit_code" else 0)

    monkeypatch.setattr(retry.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="No accepted capture"):
        retry.run_attempts(args)
    assert len(calls) == args.max_attempts
    report = json.loads((args.output / "attempts.json").read_text())
    assert report["accepted_episode"] is None
    assert not any(item["accepted"] for item in report["attempts"])
