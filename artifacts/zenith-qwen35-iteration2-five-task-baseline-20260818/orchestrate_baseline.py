"""Run the fixed five-cell Iteration-2 Zenith five-task baseline."""

import asyncio, hashlib, importlib.util, json, os, shutil, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
import tinker

ROOT = Path(__file__).resolve().parent
BROWSER = ROOT.parents[1]
SFT = ROOT.parents[2] / "sft_skills"
RL = SFT / "rl-environments"
AGENT_SOURCE = BROWSER / "scripts/run_zenith_inkling_agent.py"
WORLD = Path("/tmp/zenith-five-task-b7cbc663-linux-amd64")
MODEL = "Qwen/Qwen3.6-35B-A3B"
CHECKPOINT_ID = "67a236224c01d402695eb2952e4505cde0bf0f1cd1d7639bc751721a576e24a4"
SAMPLER = "tinker://cfb3a410-2c6b-5ed5-b711-b17b81649d01:train:0/sampler_weights/zenith-qwen35-post-update-iteration-02-20260810-sampler"
TASK_SEEDS = {task: (411001,) for task in ("zdib01", "zeko01", "zflt01", "zpal01", "zslr01")}
PACKAGE_DIGESTS = {"zdib01":"sha256:e2c9bb208f207376e4c628f7a34a014602d7e34197dbc140363d9c6f1820b9eb","zeko01":"sha256:d2cdf3df9352ec0054597c81c86d8611b2e4aabce54f5663bbb7e1a17c566033","zflt01":"sha256:5a253b604517697bdda9dcc94c866e726fae389205e869a69cbd5ae1449709de","zpal01":"sha256:7d4708459f07c6356c4e4fcff8759d5a988de1897b333c96c290a74feeecc19e","zslr01":"sha256:1ff44fd51f90fbdde7a4f895e7e024d9d5aa3e872f3761259319b5de08b9d3a3"}
WORLD_SHA = "8846f49ee0ea08cfab34cab7c1717c69ba7e2eb1a3555e1dbf519718be9b7a03"

def now(): return datetime.now(UTC).isoformat().replace("+00:00","Z")
def sha(data): return hashlib.sha256(data).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name("."+path.name+".tmp"); temp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n"); temp.replace(path)
def git(repo,*args): return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
def load_agent(task, instance="preflight"):
    name=f"zenith_agent_{task}_{instance}"; spec=importlib.util.spec_from_file_location(name,AGENT_SOURCE); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
    module.__file__=str(ROOT / "run_zenith_inkling_agent.py")
    module.TASK=f"zenith-{task}"; module.TASK_INSTRUCTION_PATH=f"harbor/tasks/zenith-{task}/instruction.md"; module.TASK_INSTRUCTION=(RL/module.TASK_INSTRUCTION_PATH).read_text().strip()
    module.SYSTEM_PROMPT="You are a browser agent. Use the browser_harness tool to achieve your goal in the already-connected local Chromium browser.\n\n"+module.BROWSER_HARNESS_SKILL
    return module
def valid(row): return not row.get("error") and row.get("cleanup",{}).get("sandbox_deleted") is True and row.get("termination_reason") != "irrecoverable_infrastructure_error"

async def main():
    for key in ("TINKER_API_KEY","DAYTONA_KEY","GOOGLE_API_KEY"):
        if not os.environ.get(key,"").strip(): raise RuntimeError(f"missing credential: {key}")
    if ROOT.joinpath("final-report.json").exists(): raise RuntimeError("final baseline already exists")
    source_head=git(RL,"rev-parse","HEAD")
    assert source_head == git(RL,"rev-parse","origin/main")
    scoped=["hosted_worlds/go_sites/zenith",*(f"harbor/tasks/zenith-{task}" for task in TASK_SEEDS)]
    if subprocess.run(["git","-C",str(RL),"diff","--quiet","b7cbc663f116fb09462864b5927c1237da6927fe",source_head,"--",*scoped]).returncode:
        raise RuntimeError("verified Zenith task or World inputs changed")
    assert sha(WORLD.read_bytes()) == WORLD_SHA
    modules={task:load_agent(task) for task in TASK_SEEDS}
    contract={"started_at":now(),"effort":"medium","experiment":"iteration2_five_task_baseline","source_head":source_head,"source_head_equals_origin_main":True,"verified_scoped_inputs_equal_expected_head":"b7cbc663f116fb09462864b5927c1237da6927fe","browser_runner_head":git(BROWSER,"rev-parse","HEAD"),"world_binary":{"path":str(WORLD),"sha256":WORLD_SHA,"static_linux_amd64":True},"task_packages":{task:{"version":"1.0.0","digest":PACKAGE_DIGESTS[task],"canonical_seed":seeds[0],"rollout_count":1,"instruction":modules[task].TASK_INSTRUCTION,"instruction_sha256":sha(modules[task].TASK_INSTRUCTION.encode())} for task,seeds in TASK_SEEDS.items()},"checkpoint":{"identity_sha256":CHECKPOINT_ID,"sampler_path":SAMPLER,"base_model":MODEL,"frozen_baseline_predates_training_client":True},"thinking":"enabled","renderer":"qwen3_5","max_generated_tokens":32000,"timeout_seconds":1200,"concurrency":5,"maximum_infrastructure_replacements_per_task":2,"valid_outcome_retries":0,"snapshot":"browser-rl-local-harness-f5eaf904-c2m4d4-v1","browser_harness_skill_sha256":"4598708be6efa99df2bd1bf517b75c9652c9bca77372e2bfbdfa5359c8d9be3d","agent_source_sha256":"a6c20b8a4ee5285ce5dcb4aea872fd43cca895c47fc1eff1a007b5cc77b4eb26","verifier_source_sha256":"daef038ca0bf43f59f8196ae2c35abb8cb518e9d40dd0c3d6dc67c5a823b5db1","sole_numeric_reward":"canonical_internal_v2","deterministic_qc_scoring":False,"training_calls":0,"optimizer_steps":0,"browser_use_cloud":False}
    write(ROOT/"predispatch-contract.json",contract)
    preflight_dir=ROOT/"preflight-agent-loop"; await modules["zslr01"].validate_loop(preflight_dir)
    write(ROOT/"preflight.json",{"validated_at":now(),"zero_provider_calls":True,"zero_sandboxes":True,"tasks":list(TASK_SEEDS),"cells":5,"canonical_world_seed":411001,"world_sha256":WORLD_SHA,"checkpoint_identity_sha256":CHECKPOINT_ID,"package_digests":PACKAGE_DIGESTS,"agent_loop_validation":str(preflight_dir/"validation.json")})
    service=tinker.ServiceClient(user_metadata={"recipe":"zenith_iteration2_five_task_baseline"}); client=await service.create_sampling_client_async(model_path=SAMPLER)
    semaphore=asyncio.Semaphore(5); state={"active":0,"peak":0}; lock=asyncio.Lock()
    async def run_one(task,ordinal,seed,attempt=1):
        module=load_agent(task, f"{ordinal}_{seed}_{attempt}"); module.WORLD_SEED=seed; task_root=ROOT/"tasks"/task; task_root.mkdir(parents=True,exist_ok=True)
        async with semaphore:
            async with lock: state["active"]+=1; state["peak"]=max(state["peak"],state["active"])
            try:
                renderer_name,renderer=module.renderer_for_model(MODEL,"enabled")
                args=SimpleNamespace(world_binary=WORLD,customer_repo=RL,model=MODEL,thinking="enabled",max_tokens=32000,timeout=1200)
                return await module.run_rollout(args,task_root,ordinal,seed,client,renderer,renderer_name)
            except Exception as exc:
                rollout=task_root/f"rollout-{ordinal:02d}"; cleanup=rollout/"execution/cleanup.json"
                row={"task":task,"ordinal":ordinal,"sampling_seed":seed,"attempt":attempt,"error":f"{type(exc).__name__}: {exc}","cleanup":json.loads(cleanup.read_text()) if cleanup.exists() else None}
                write(rollout/"summary.json",row); return row
            finally:
                async with lock: state["active"]-=1
    rows=await asyncio.gather(*(run_one(task,i,seed) for task,seeds in TASK_SEEDS.items() for i,seed in enumerate(seeds,1)))
    final=[]
    for row in rows:
        if valid(row): final.append(row); continue
        task,ordinal,seed=row["task"],row["ordinal"],row["sampling_seed"]
        invalid_root=ROOT/"invalid-attempts"/task/f"rollout-{ordinal:02d}-attempt-01"; invalid_root.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(ROOT/"tasks"/task/f"rollout-{ordinal:02d}"),invalid_root)
        replacement=await run_one(task,ordinal,seed,2)
        if not valid(replacement):
            second=ROOT/"invalid-attempts"/task/f"rollout-{ordinal:02d}-attempt-02"; shutil.move(str(ROOT/"tasks"/task/f"rollout-{ordinal:02d}"),second); replacement=await run_one(task,ordinal,seed,3)
        if not valid(replacement): raise RuntimeError(f"replacement ceiling exhausted: {task} seed {seed}")
        final.append(replacement)
    assert len(final)==5 and all(valid(r) for r in final)
    write(ROOT/"rollout-cohort-receipt.json",{"completed_at":now(),"valid_cells":5,"peak_concurrency":state["peak"],"rows":final})
    proc=await asyncio.create_subprocess_exec(str(SFT/"tools/internal_v2_judge/.venv/bin/python"),str(ROOT/"judge_batch.py"),"--root",str(ROOT),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT,env=os.environ.copy()); output,_=await proc.communicate(); (ROOT/"internal-v2.log").write_bytes(output)
    if proc.returncode: raise RuntimeError("Internal V2 batch failed: "+output.decode(errors="replace")[-4000:])
    batch=json.loads((ROOT/"internal-v2-batch-receipt.json").read_text()); assert batch["calls"]==5
    per_task={}
    for task in TASK_SEEDS:
        task_rows=sorted((r for r in batch["rows"] if r["task"]==task),key=lambda x:x["ordinal"]); rewards=[float(r["reward"]) for r in task_rows]
        per_task[task]={"canonical_seed":411001,"rewards":rewards,"mean":sum(rewards)/len(rewards),"strict_successes":sum(bool(r["strict_pass"]) for r in task_rows),"failure_categories":[r["failure_category"] for r in task_rows if not r["strict_pass"]]}
    macro=sum(x["mean"] for x in per_task.values())/5; micro=sum(sum(x["rewards"]) for x in per_task.values())/5; assert abs(macro-micro)<1e-12
    usage={"agent_generated_tokens":sum(int(r["usage"]["generated_tokens"]) for r in final),"agent_prompt_tokens_sum":sum(int(r["usage"]["prompt_tokens_sum"]) for r in final),"agent_cost_exposed":False,"judge_usage":[json.loads((ROOT/"tasks"/r["task"]/f"rollout-{r['ordinal']:02d}"/"internal-v2/receipt.json").read_text()).get("usage") for r in batch["rows"]],"judge_cost":[json.loads((ROOT/"tasks"/r["task"]/f"rollout-{r['ordinal']:02d}"/"internal-v2/receipt.json").read_text()).get("cost") for r in batch["rows"]]}
    report={"completed_at":now(),"experiment":"iteration2_five_task_baseline","checkpoint_identity_sha256":CHECKPOINT_ID,"per_task":per_task,"macro_mean":macro,"micro_mean":micro,"macro_micro_equal":True,"macro_strict_rate":sum(x["strict_successes"] for x in per_task.values())/5,"valid_cells":5,"judge_calls":5,"usage":usage,"training_calls":0,"optimizer_steps":0}
    write(ROOT/"final-report.json",report); print(json.dumps(report,indent=2))
if __name__=="__main__": asyncio.run(main())
