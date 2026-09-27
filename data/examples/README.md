# Example clips

Each subset ships one short clip together with the exact QA rows that
reference it, so the example is self-contained. The clip is the full
episode window; the sibling `<episode>.questions.jsonl` lists every
question whose `query_time` falls inside that window, in the same schema
as `data/annotations/`. The complete question sets live in
`data/annotations/<subset>.questions.jsonl`.

| subset | example clip | episode_id | QA rows | query_time range (s) |
| --- | --- | --- | --- | --- |
| `storm_cook` | `P02-20240210-220650_revisit.mp4` | `P02-20240210-220650_revisit` | 11 | 2–8 |
| `storm_bike` | `indiana_bike_06_window007_revisit.mp4` | `indiana_bike_06_window007_revisit` | 12 | 3–16 |
| `storm_healthy` | `utokyo_pcr_2001_25_window003_revisit.mp4` | `utokyo_pcr_2001_25_window003_revisit` | 9 | 3–18 |
| `storm_music` | `iiith_piano_002_window001_revisit.mp4` | `iiith_piano_002_window001_revisit` | 9 | 7–43 |
| `storm_sports` | `unc_basketball_02-24-23_01_window005_revisit.mp4` | `unc_basketball_02-24-23_01_window005_revisit` | 9 | 0–9 |
| `storm_sim` | `STORM_FloorPlan406_seed323460856_96a9e1cc113f.mp4` | `STORM_FloorPlan406_seed323460856_96a9e1cc113f` | 11 | 17–51 |
| `storm_vhome` | `scene_5_room_162_seed_20263304.mp4` | `scene_5_room_162_seed_20263304` | 10 | 6.05–64.05 |
