# Tasks

Task manifests are JSON Lines files. Each row contains:

```json
{"task_id":"stable-id","split":"train","instruction":"...","start_url":"https://...","grader":"world-v1"}
```

Use distinct `train` and `held_out` rows. Never report training-set performance as held-out
improvement.

