#!/usr/bin/env python3
"""Rebuild v2.5 QA from a 28-video dataset and its rollout traces."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main(argv=None):
    from tools.storm.benchgen.stream_eqa import balanced_natural_evidence_qa_exp2 as profile
    args = profile.parse_args(argv)
    profile._configure_base()
    # Replace the historical report so old model scores are not copied into new exports.
    profile.base._report_markdown = lambda summary, source, output: (
        '# Rule QA export\n\n'
        'Profile: balanced-v25 (original v2.5 experiment 2 rules).\n\n'
        f'Source: `{source}`\n\nOutput: `{output}`\n\n'
        'Static checks passed. This new export has NOT been evaluated with a VLM.\n'
        'Run your own semantic review and text/video evaluation after polishing.\n\n'
        f'Questions: {summary["question_count"]}; videos: {summary["video_count"]}.\n'
    )
    summary = profile.base.validate_export(args.output) if args.validate_only else profile.base.export_clone(
        args.source_dataset, args.output, link_mode=args.link_mode, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
