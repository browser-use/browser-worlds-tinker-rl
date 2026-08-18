"""One canonical Internal V2 call for each valid five-task baseline trace."""

import argparse, asyncio, base64, hashlib, json, sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SFT = HERE.parents[2] / "sft_skills"
RL = SFT / "rl-environments"
JUDGE_ROOT = SFT / "tools/internal_v2_judge"
sys.path.insert(0, str(JUDGE_ROOT / "src"))
from browser_use import ChatGoogle
from internal_v2_judge import InternalV2Judge, InternalV2JudgementResult, RubricState

TASKS = ("zdib01", "zeko01", "zflt01", "zpal01", "zslr01")
PRESERVED_ROLLOUTS = {
    "zflt01": HERE / "invalid-attempts/zflt01/rollout-01-attempt-02",
}
RUBRICS = {
    "zdib01": (RL / "docs-site/data/rubrics/zdib01.md", "a5f9d2bce6bf91916ad955425f384011a0b6965a3a482e39789f5f5f3a1d69a7"),
    "zeko01": (RL / "docs-site/data/rubrics/zeko01.md", "ef4d175b49b8726c2cea4298d287fe93d8bb7ddd3873d4c47f9f6878c05eeaf5"),
    "zflt01": (RL / "docs-site/data/rubrics/zflt01.md", "7742e567e267df81628ec391fa355a45a80d313449afd072956378f3014937f1"),
    "zpal01": (JUDGE_ROOT / "rubrics/zpal01.md", "4293d8ca8a5e61b4f96f632aaf8e4d1a80cab54f9fb281c1f0170a401c8bc765"),
    "zslr01": (RL / "docs-site/data/rubrics/zslr01.md", "24964a462c63e95c5542a5fa5a05746611c80a7760b9e004cf8a4d039071e143"),
}
JUDGE_SOURCE_SHA = "8415f81b79533b5fdf978f85c6775c6bbb77b6a9ad2abafb87d9b56710c4f4e9"
JUDGE_PROMPT_SHA = "5d3f5f25dcf9205b001134b1959ed3e3a16104ab8a563e2a56034e84bdae252d"
JUDGE_REPO_HEAD = "604fc0ee092d630de23b13534861c153d9a06178"

def sha(data): return hashlib.sha256(data).hexdigest()
def now(): return datetime.now(UTC).isoformat().replace("+00:00", "Z")
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(path)
def content_text(content):
    if isinstance(content, str): return content
    if not isinstance(content, list): return str(content or "")
    return "\n".join(str(x.get("thinking") or x.get("text") or "") if isinstance(x, dict) else str(x) for x in content)
def projection(rollout):
    trace = json.loads((rollout / "trace.json").read_text())
    ready = json.loads((rollout / "initial-grounding.json").read_text())
    steps = ["browser_execute: initial grounded browser observation\nresult: " + str(ready["observation"])]
    images, image_steps = [rollout / "execution/initial.png"], [1]
    events = trace["events"]
    for ix, event in enumerate(events):
        if event["type"] == "assistant":
            content = content_text(event["message"].get("content"))
            if content.strip(): steps.append("thinking: " + content)
        elif event["type"] == "tool_result":
            prior = next(e for e in reversed(events[:ix]) if e["type"] == "assistant" and e["turn"] == event["turn"])
            call = (prior["message"].get("tool_calls") or [])[event["call"] - 1]
            code = json.loads(call["function"]["arguments"])["code"]
            steps.append(f"browser_execute: {code}\n{'error' if event.get('is_error') else 'result'}: {event['content']}")
            shot = rollout / f"execution/turn-{event['turn']:04d}-call-{event['call']:02d}.png"
            if shot.is_file(): images.append(shot); image_steps.append(len(steps))
        elif event["type"] == "error_feedback": steps.append("error_feedback: " + event["content"])
    outputs, out_dir = [], rollout / "execution/outputs"
    if out_dir.is_dir():
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            data = path.read_bytes()
            try: text = data.decode() if len(data) <= 300000 else None
            except UnicodeDecodeError: text = None
            outputs.append({"name": str(path.relative_to(out_dir)), "size": len(data), "sha256": sha(data), "text": text})
    return trace, steps, images, image_steps, outputs

async def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); llm = ChatGoogle(model="gemini-3-flash-preview", temperature=0, thinking_budget=1024, max_output_tokens=8096)
    semaphore = asyncio.Semaphore(5)
    async def one(task, ordinal):
        async with semaphore:
            current = args.root / "tasks" / task / f"rollout-{ordinal:02d}"
            rollout = PRESERVED_ROLLOUTS.get(task, current)
            summary = json.loads((rollout / "summary.json").read_text())
            trace, steps, images, image_steps, outputs = projection(rollout)
            instruction = (RL / f"harbor/tasks/zenith-{task}/instruction.md").read_text().strip()
            rubric_path, rubric_sha = RUBRICS[task]; rubric_content = rubric_path.read_text()
            assert sha(rubric_content.encode()) == rubric_sha
            result = SimpleNamespace(final_result=trace["final_response"], steps=steps, screenshots_b64=[base64.b64encode(p.read_bytes()).decode() for p in images], screenshot_steps=image_steps, output_files=outputs)
            meta = {}
            async def evaluate(messages, schema, model):
                assert schema is InternalV2JudgementResult and model == "gemini-3-flash-preview"
                meta["judge_messages_sha256"] = sha(json.dumps([m.model_dump(mode="json") for m in messages], sort_keys=True, default=str).encode())
                response = await llm.ainvoke(messages, output_format=schema)
                meta["usage"] = getattr(response, "usage", None); meta["cost"] = getattr(response, "cost", None)
                return response.completion
            evaluation = await InternalV2Judge.evaluate(task=instruction, result=result, task_data={"website":"zenith"}, task_id=task, rubric_state=RubricState(content=rubric_content, commit=rubric_sha), evaluate_task=evaluate)
            judgement = evaluation.judgement; reward = float(InternalV2Judge.score({}, result, judgement))
            receipt = {"judged_at":now(), "single_call":True, "task":task, "ordinal":ordinal, "seed":summary["sampling_seed"], "reward":reward, "strict_pass":bool(judgement.verdict), "raw_judgement":judgement.model_dump(mode="json"), "judge_model":"gemini-3-flash-preview", "judge_repo_head":JUDGE_REPO_HEAD, "judge_source_sha256":JUDGE_SOURCE_SHA, "judge_prompt_sha256":JUDGE_PROMPT_SHA, "rubric_path":str(rubric_path), "rubric_sha256":rubric_sha, "instruction_sha256":sha(instruction.encode()), "trace_sha256":sha((rollout/"trace.json").read_bytes()), "final_response_sha256":sha(trace["final_response"].encode()), "screenshots":[{"path":str(p.relative_to(rollout)),"step":s,"sha256":sha(p.read_bytes())} for p,s in zip(images,image_steps)], "output_files":outputs, **meta}
            write(rollout / "internal-v2/receipt.json", receipt)
            return {"task":task,"ordinal":ordinal,"seed":summary["sampling_seed"],"reward":reward,"strict_pass":bool(judgement.verdict),"failure_category":judgement.failure_category,"rollout_path":str(rollout)}
    rows = await asyncio.gather(*(one(task, 1) for task in TASKS))
    write(args.root / "internal-v2-batch-receipt.json", {"judged_at":now(),"calls":len(rows),"one_call_per_valid_trace":True,"rows":rows,"judge_repo_head":JUDGE_REPO_HEAD})
    print(json.dumps(rows))
if __name__ == "__main__": asyncio.run(main())
