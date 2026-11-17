# RevisitQA answerability labeling rubric

Use this rubric to decide, for one question at its query time `t_q`, whether the video **up to and including `t_q`**
determines the answer (**Known**) or not (**Uncertain**), and why. You see only frames at or before `t_q`
(1 frame per second, the frame at `t_q` included). Never use knowledge of what happens after `t_q`.

## 1. Known vs Uncertain

A question is **Known** when exactly one of the three candidate answers is supported by the observed frames and the
other two are ruled out. Otherwise it is **Uncertain**.

Apply these rules in order:

1. **Directly visible.** The fact is visible in one or more frames ≤ `t_q` → Known.
2. **Inferred from observed states.** The before-state and after-state are both observed, and the change between them
   is unambiguous, even if the motion itself was not filmed (e.g., a pitcher seen on the counter at 35 s and on the
   stove at 38 s "was moved from the counter to the stove") → Known. Do **not** require seeing the action itself.
3. **Questions restricted to what is shown.** If the question asks about what is *visible / shown / in the frames*
   (e.g., "how many visible pouring actions are shown"), answer from the frames as given; events that may have
   happened off-screen do not matter → normally Known.
4. **Questions about what actually happened.** If the question asks how many times something happened, which happened
   first, or what caused a change, *without* restricting to what is shown, and an unobserved gap (black frame, camera
   away, occlusion) could hide a relevant event that would change the answer → Uncertain.
5. **Current state ("by now", "currently", "at this moment").** The state must be observed at or after the last moment
   it could have changed. If the object was last seen before a gap or before the camera moved away, and the candidates
   differ in what may have happened since, an earlier observation is **stale** → Uncertain. Do not answer a current-state
   question from an outdated view.
6. **Identity across a gap.** "Is it the same object" is Known only if a distinguishing mark, continuous tracking, or
   the absence of any look-alike candidate settles it; otherwise Uncertain.
7. **Readability at evaluation resolution.** Models see each frame at about 448×448 pixels (200,704-pixel budget).
   Text, numbers, logos and small objects are Known only if legible at that resolution, not merely in the original video.
   Judge this on the provided evaluation-resolution frame, never on grid thumbnails. Before answering "cannot see",
   check the original-resolution frame of the query time: a thumbnail that hides an object is not evidence that the
   video hides it. Make sure you are looking at the object the question names.
8. **Candidate check.** If two candidates both remain consistent with the frames → Uncertain. If none fits → Uncertain
   (note "no candidate fits" in the evidence).

9. **Invalid item.** Mark `"status": "invalid"` (with a reason in `evidence`) when the question cannot be answered as
   posed by any video, e.g., it asks what is visible inside a black frame, its candidates contradict the facts that
   the question itself presupposes, or two candidates are equivalent.

## 2. Uncertainty causes (only for Uncertain; choose a primary cause, optionally others)

- `missing_observation`: the decisive moment/process is not in view (black frame, camera away, not yet happened, stale current state).
- `partial_observation`: only part is visible or it is occluded.
- `ambiguous_evidence`: visible but consistent with several readings.
- `low_visual_quality`: blur, too small, unreadable text.
- `multiple_candidates`: several same-category objects; the referent cannot be fixed.
- `ambiguous_attribute`: color, material, or similar attribute cannot be decided.

## 3. Output (one JSON line per item)

```json
{"id": "...", "status": "known", "choice": "A", "causes": [], "rule": "2", "evidence": "t=35s pitcher on counter; t=38s on stove burner"}
{"id": "...", "status": "uncertain", "choice": null, "causes": ["missing_observation"], "rule": "5", "evidence": "last seen t=23s by the window; camera away from t=30s; t_q=45s"}
```

`status` is `known`, `uncertain`, or `invalid`. `choice` is the letter of the supported substantive option (A–C) for Known items. `rule` is the number of the rule
in Section 1 that decided the label. `evidence` cites the frame times you relied on.
