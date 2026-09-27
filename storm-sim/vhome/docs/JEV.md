# Optional Jev judgments

QA polishing runs after capture and does not block room workers. Jev requests
can run on a separate host from the simulator. API credentials are read from
the process environment.

## Useful boundaries

| Location | Existing logic | Suitable Jev judgment | Code still owns |
| --- | --- | --- | --- |
| `scripts/rollout_rooms_native.py` | Error substring matching and target exclusions | Classify unfamiliar log symptoms with an unknown option | Exact known errors, process ownership, retry budgets, target IDs |
| `src/storm_virtualhome/room_program.py` | Verb coverage, bounding-box size and distance ranking | Compare the household plausibility of eligible candidate routines | Seeds, dependencies, actor roles, reachability and event windows |
| `src/storm_virtualhome/room_enrichment.py` | Room/prop class whitelist | Rank appropriate props from a native candidate pool | Asset identity, bounding boxes, support, collision and replay checks |
| `scripts/polish_qa.py` | Checks only the rewrite response shape | Verify object, temporal and epistemic meaning after a rewrite | Labels, options, timestamps, video prefixes and unchanged protected fields |
| `src/storm_virtualhome/natural_qa.py` | Template and option-domain construction | Flag semantic option overlap and explicit wording cues | Answer derivation, choice counts, chance baselines and balanced sampling |

The first version implements standalone failure, plan and QA reviews plus an
optional wording-preservation check in the VLM adapter. Plan judgments are advisory;
they do not automatically change frozen programs. Known error checks should become
structured error codes, rather than being replaced by API calls.

Jev returns typed decisions, not rewritten text. VLM generation remains separate.
Numerical checks, native annotations, physics, segmentation and frame continuity
stay deterministic. Text-only wording review cannot establish visual correctness,
text-only benchmark accuracy, or a calibrated chance baseline.

## Run locally

Set `TYPESAFE_API_KEY` in the process environment. The adapter uses the documented
`https://api.typesafe.ai/v1/systemone` endpoint. No additional package is needed.

```bash
python scripts/review_jev.py --kind failure --input recovery_status.json \
  --output reports/failure_review --workers 4
python scripts/review_jev.py --kind plan --input frozen_programs.json \
  --output reports/plan_review --workers 4
python scripts/review_jev.py --kind qa --input revised/questions.jsonl \
  --original original/questions.jsonl --output reports/qa_review --workers 4
```

The QA input accepts an episode `qa.json`, a JSON array or JSONL. A plan input is
one event program or an array of programs, with `room` or `room_name` supplied.
Failure input accepts `recovery_status.json` or an array of failure records.
`--dry-run` writes requests without sending them. Output directories must be new.
All reports retain request hashes, complete requests, model versions, raw answers
and elapsed time. API errors produce `unavailable` and a nonzero CLI exit status.
There are no automatic API retries or cross-run caches in this version.

For a later VLM wording pass, add `--jev-check` to `scripts/polish_qa.py`. The
original wording is kept unless all three preservation judgments return
`preserved` with confidence at least 0.8. This is a conservative starting policy,
not a calibrated accuracy claim. Other wording diagnostics remain advisory.

QA review state is an allowlist of question text, options and original wording.
It excludes answer indices, hidden events, private evidence and future state.
With `--original`, protected fields and the set of IDs must match before API calls.

## Local smoke check, 2026-09-21

51 live calls to resolved model `jev-1.13.0` returned valid typed responses:
17 real failure records, 12 frozen plans, 12 unchanged historical questions,
4 deliberately altered questions and 6 hand-written paraphrases. Median request
latency was 1.10 seconds. Requests ran with four workers per command.

All four planted semantic changes were rejected by the preservation policy;
12 unchanged controls and 4 of 6 paraphrases were supported. Two paraphrases were
left for review. This small smoke check is not a dataset accuracy study and does
not demonstrate improved rollout throughput. Some failure labels had low
confidence, so no model output was used to exclude targets or alter acceptance.

Read the [HTTP API](https://docs.typesafe.ai/api.md),
[confidence guide](https://docs.typesafe.ai/confidence.md), and
[evidence verification example](https://docs.typesafe.ai/cookbooks/citation_check.md)
when extending the adapter. Noul is the probability of yes, not a graded quality
score and not a separate confidence value. Evaluate candidate ranking on held-out
rooms before using it to prioritize rendering.
