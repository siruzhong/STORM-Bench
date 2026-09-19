# Public schema

Question records are JSON objects, one per line. Required fields are `id`, `episode_id`, `query_time`, `question_type`, `question`, `options` (four strings), `answer_index` (0--3), `diagnostics.epistemic_status` (`known`/`uncertain`), `diagnostics.uncertainty_sources`, and `change_intensity` (1--10). Optional `evidence_spans` must satisfy `0 <= start <= end <= query_time`.

Prediction records contain `id`, `task_answer` (0--3 or A--D), `status`, and optional `uncertainty_sources`. For uncertain questions, option D (index 3) is the designated evidence-insufficiency answer.

The evaluator computes overall task accuracy, status and joint accuracy, known/uncertain diagnostics, overconfidence, uncertainty-source F1, and Laplace-smoothed `Storm-BR`. Intensity bins are B1=`1..3`, B2=`4..6`, and B3=`7..10`; empty cells are excluded.
