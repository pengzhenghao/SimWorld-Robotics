# SimWorld-Robotics Single-Agent Baseline: reproduce paper-style metrics

This folder contains:
- `single_exp.py`: interactive/single-task runner (LLM in the loop)
- `metric.py`: **paper-style metrics** computed from `trajectory.csv` + `task_config.json`
- `evaluate.py`: CLI aggregator over many `trajectory.csv` episodes (prints SR/SPL/Subtask SR/Distance progress + collisions/violations)
- `sweep_exp.py`: run a chosen LLM over **all tasks** in a split (produces `trajectory.csv` for every episode)

The SimWorld-Robotics paper reports Success Rate (SR), Subtask SR, Distance Progress, and collision/violation metrics; in this code release those are implemented in `metric.py` and are reproducible from the environment logs (see [`SimWorld_Robotics.pdf`](https://simworld.org/assets/SimWorld_Robotics.pdf)).

---

## 0) Prereqs

You must have:
- Unreal Engine + the SimWorld UE project running (UnrealCV server)
- `simworld_gym` installed (`pip install -e external/SimWorld-Robotics/simworld_gym`)
- The test settings extracted under:
  - `simworld_gym/SimWorldGym/simworld_gym/envs/setting/single_agent_world/`

Per repo README, the official test settings tarballs are:
- `single_test.tar.gz` → `.../single_agent_world/`
- `multi_test.tar.gz` → `.../multi_agent_world/`

---

## 1) Dataset “splits” in this repo

Under `.../envs/setting/single_agent_world/` you typically have:
- `easy/`: the main single-agent benchmark split in this release (many maps)
- `sample/`: tiny demo subset (few tasks)

Inside each `easy/map_road_XX_YY/` there are usually two tasks:
- `task_dist_11_0_1`
- `task_dist_21_0_1`

These two task IDs correspond to different route distances (shorter vs longer), and are convenient knobs for “easier vs harder” within the `easy/` split.

> Note: this code release does **not** ship a separate `hard/` directory under `single_agent_world/`. If your paper setup uses an explicit hard split, you’ll need the corresponding settings tarball (not included here).

---

## 2) Run one task (quick sanity)

From this directory:

```bash
cd external/SimWorld-Robotics/simworld_gym/baseline/single

# Required for Gemini backend
export GEMINI_API_KEY=...

UE_PORT=9000 python single_exp.py \
  --backend gemini \
  --model gemini-2.5-flash \
  --map 20_0 \
  --setting simple
```

This will run a single task (`task_dist_11_0_1` or `task_dist_21_0_1` depending on the script logic) and will:
- interact with the environment using discrete actions
- produce environment logs (including `trajectory.csv`) under the run’s `log_dir`

---

## 3) Sweep the full dataset (easy split) with a chosen model

This is the “paper reproduction” mode: run through **all tasks** in a split and write `trajectory.csv` logs for each episode.

```bash
cd external/SimWorld-Robotics/simworld_gym/baseline/single
export GEMINI_API_KEY=...

UE_PORT=9000 python sweep_exp.py \
  --backend gemini \
  --model gemini-2.5-flash \
  --env simple \
  --split easy \
  --out_dir /abs/path/to/runs/gemini-2.5-flash_easy_simple \
  --tasks task_dist_11_0_1 task_dist_21_0_1
```

Notes:
- `--out_dir` should be **absolute**. The simulator will write per-episode logs under:
  - `<out_dir>/logs/simpleenv_*/reset_*/trajectory.csv` (SimpleEnv)
  - `<out_dir>/logs/trafficenv_*/reset_*/trajectory.csv` (TrafficEnv)
- By default the env creates an `images/` folder per episode but does **not** populate it. To save per-step images/videos, pass `--record_video` (can be large).

---

## 4) Aggregate and report full metrics (SR/Subtask SR/Distance Progress/Collisions/Violations)

Once you have logs, run:

```bash
cd external/SimWorld-Robotics/simworld_gym/baseline/single

# SimpleEnv metrics
python evaluate.py \
  --run_dir /abs/path/to/runs/gemini-2.5-flash_easy_simple \
  --env simpleenv \
  --split easy \
  --with_collisions

# TrafficEnv metrics (adds red-light violations)
python evaluate.py \
  --run_dir /abs/path/to/runs/gemini-2.5-flash_easy_traffic \
  --env trafficenv \
  --split easy \
  --with_collisions \
  --with_violations
```

What each metric means (from `metric.py`):
- **SR**: fraction of episodes with `success==True` at the final row (implemented as `SPL>0`)
- **SPL**: `min(ideal_distance / moving_distance, 1)` if successful
- **Subtask_SR**: `subtask_success / len(instruction)`
- **Distance_Progress**: how much the closest approach reduced distance-to-goal vs start
- **Static_Collisions**: sum of `object_collision + building_collision`
- **Dynamic_Collisions**: sum of `human_collision + vehicle_collision`
- **RedLight_Violations**: sum of `red_light_violation` (TrafficEnv only)

---

## 5) Common gotchas

- **No `trajectory.csv` found**: make sure you passed an absolute `--out_dir` to `sweep_exp.py` and that the environment successfully ran through at least one episode.
- **LLM output parsing failures**: the baseline expects strict JSON from the model; if you see parse errors, try a model with stronger structured-output behavior or reduce temperature.
- **Traffic metrics**: TrafficEnv requires traffic settings; this repo release may not include per-task `traffic.json` under `single_agent_world/easy/` by default.


