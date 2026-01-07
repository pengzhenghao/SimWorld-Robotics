from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import gym
import numpy as np
import simworld_gym

from agents import ReasoningAgent, ReActAgent
from baseline_utils import action_history_text, numpy_to_base64, split_into_strips
from prompt_template import nav_template, perception_template, reasoning_template


def _iter_tasks(split: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (map_dir, task_dir) pairs from the installed simworld_gym settings.
    Example:
      ("map_road_20_0", "task_dist_11_0_1")
    """
    gympath = os.path.dirname(simworld_gym.__file__)
    base = os.path.join(gympath, "envs", "setting", "single_agent_world", split)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Split not found: {base}")
    for map_dir in sorted(os.listdir(base)):
        map_path = os.path.join(base, map_dir)
        if not (os.path.isdir(map_path) and map_dir.startswith("map_road_")):
            continue
        for task_dir in sorted(os.listdir(map_path)):
            task_path = os.path.join(map_path, task_dir)
            if os.path.isdir(task_path) and task_dir.startswith("task_"):
                yield map_dir, task_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--env", choices=["simple", "traffic"], default="simple")
    parser.add_argument("--split", choices=["easy", "sample"], default="easy")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task dirs to include (e.g. task_dist_11_0_1)")
    parser.add_argument("--maps", nargs="*", default=None, help="Optional map dirs to include (e.g. map_road_20_0)")
    parser.add_argument(
        "--max_maps",
        type=int,
        default=0,
        help="Maximum number of unique maps to run (<=0 means all). Default: 10.",
    )
    parser.add_argument("--ue_port", type=int, default=int(os.getenv("UE_PORT", "9000")))
    parser.add_argument("--out_dir", type=str, required=True, help="Absolute output dir for env logs (will contain ./logs/...)")
    parser.add_argument("--max_episodes", type=int, default=-1, help="Stop after N episodes (<=0 means all)")
    parser.add_argument("--strip", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument("--reasoning", action="store_true", help="Use ReasoningAgent instead of ReActAgent")
    parser.add_argument("--log_every_steps", type=int, default=25, help="Print one line every N env steps. Default: 25.")
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="If set, SimWorldGym will save per-step images/videos under each episode's images/ directory (can be large).",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Gym still references np.bool8 in some versions; numpy 2.x removed it
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_

    env_id = "simworld_gym/SimpleWorld" if args.env == "simple" else "simworld_gym/TrafficWorld"

    # This log_dir is used by the env to create <log_dir>/logs/<env>_*/reset_*/trajectory.csv
    env = gym.make(
        env_id,
        port=args.ue_port,
        resolution=(720, 600),
        render_mode="rgb_array",
        observation_type="all",
        record_video=args.record_video,
        log_dir=out_dir,
        reward_setting={
            "human_collision_penalty": -1,
            "object_collision_penalty": -0.1,
            "action_penalty": -0.1,
            "success_reward": 10.0,
            "off_track_penalty": -0.01,
        },
    )

    if args.reasoning:
        agent = ReasoningAgent(
            backend=args.backend,
            model=args.model,
            system_prompt=nav_template(strip=args.strip, depth=args.depth, segment=args.segment),
        )
    else:
        agent = ReActAgent(
            backend=args.backend,
            model=args.model,
            reasoning_prompt=reasoning_template(strip=args.strip),
            perception_prompt=perception_template(strip=args.strip, depth=args.depth, segment=args.segment),
        )

    ACTION_TO_STEP = {0: 0, 1: 5, 2: 4, 3: 2, 4: 3}
    action_mapping = ["Move_Forward", "Rotate_Left", "Rotate_Right", "Move_Left", "Move_Right", "Subtask_Completed"]

    # Build episode list (map/task pairs), with optional filtering + max_maps.
    pairs: List[Tuple[str, str]] = []
    seen_maps: List[str] = []
    for map_dir, task_dir in _iter_tasks(args.split):
        if args.maps and map_dir not in args.maps:
            continue
        if args.tasks and task_dir not in args.tasks:
            continue
        if args.max_maps > 0 and map_dir not in seen_maps:
            if len(seen_maps) >= args.max_maps:
                continue
            seen_maps.append(map_dir)
        pairs.append((map_dir, task_dir))

    if not pairs:
        raise SystemExit("No episodes matched your filters (split/maps/tasks/max_maps).")

    # Optional tqdm progress bar
    try:
        from tqdm import tqdm  # type: ignore

        it = tqdm(pairs, desc="episodes", unit="ep")
        use_bar = True
    except Exception:
        it = pairs
        use_bar = False

    episodes_run = 0
    episodes_ok = 0
    episodes_failed = 0
    total_env_steps = 0
    total_llm_failures = 0
    t0 = time.time()

    print(
        f"[sweep] env={args.env} split={args.split} backend={args.backend} model={args.model} "
        f"episodes={len(pairs)} max_maps={args.max_maps} out_dir={out_dir}"
    )

    for idx, (map_dir, task_dir) in enumerate(it, start=1):

        task_path = os.path.join("single_agent_world", args.split, map_dir, task_dir)
        options = {
            "task_path": task_path,
            "agent_json": os.path.join(task_path, "task_config.json"),
            "world_json": os.path.join(task_path, "progen_world.json"),
        }

        if not use_bar:
            print(f"[episode {idx}/{len(pairs)}] map={map_dir} task={task_dir}")
        agent.reset_state()
        observation, info = env.reset(options=options)

        vision_cue = info["current_instruction"]["image"]
        instruction = info["current_instruction"]["text"]
        chosen_actions: List[int] = []
        action_history: List[int] = []

        terminated = False
        parse_failure = 0
        step_i = 0
        ep_start = time.time()

        while not terminated:
            orientation = info["agent"]["agent_rotation"]

            # Build model multimodal inputs: current view + expected view (+ optional aux)
            images = []
            if args.strip:
                strips = split_into_strips(observation["rgb"])
                for idx, img in enumerate(strips):
                    images.append(
                        {
                            "img": numpy_to_base64(img),
                            "description": ["The view on the left", "The horizontal center", "The right"][idx],
                        }
                    )
                images.append({"img": numpy_to_base64(vision_cue), "description": "The expected view"})
            else:
                images.append({"img": numpy_to_base64(observation["rgb"]), "description": "The current view"})
                images.append({"img": numpy_to_base64(vision_cue), "description": "The expected view"})
            if args.segment:
                images.append(
                    {"img": numpy_to_base64(observation["object_mask"]), "description": "The object segmentation mask of the current view"}
                )
            if args.depth:
                images.append({"img": numpy_to_base64(observation["depth_map"]), "description": "The depth map of the current view"})

            action_history_str = action_history_text(action_history, action_mapping)

            try:
                result = agent.step(
                    observation=images,
                    instruction=instruction,
                    orientation=orientation,
                    action_history_text=action_history_str,
                    chosen_actions=chosen_actions,
                )
                chosen_actions = result.get("actions") or []
                parse_failure = 0
            except Exception as e:
                parse_failure += 1
                total_llm_failures += 1
                print(f"[warn] agent step failed (count={parse_failure}): {e}")
                if parse_failure > 10:
                    print("[abort] too many agent failures")
                    break
                continue

            if not chosen_actions:
                print("[abort] model returned empty actions")
                break

            for a in chosen_actions:
                action_history.append(a)
                step_i += 1
                total_env_steps += 1
                if args.log_every_steps > 0 and (step_i % args.log_every_steps == 0):
                    cur_instr = (instruction or "").replace("\n", " ")
                    print(
                        f"[step] map={map_dir} task={task_dir} step={step_i} action={a} "
                        f"orientation={orientation} instr='{cur_instr[:120]}'"
                    )
                if a == -1:
                    observation, _, terminated, _, info = env.step(-1)
                    if terminated:
                        break
                    instruction = info["current_instruction"]["text"]
                    vision_cue = info["current_instruction"]["image"]
                    action_history = []
                else:
                    env_act = ACTION_TO_STEP.get(a)
                    if env_act is None:
                        continue
                    observation, _, terminated, _, info = env.step(env_act)
                if terminated:
                    break

        episodes_run += 1
        elapsed = time.time() - ep_start
        # env returns info["success"] even before episode ends; final termination indicates completion/timeout
        ep_success = bool(info.get("success", False)) if isinstance(info, dict) else False
        if ep_success:
            episodes_ok += 1
        else:
            episodes_failed += 1
        print(
            f"[episode_done] map={map_dir} task={task_dir} steps={step_i} success={ep_success} "
            f"llm_failures={parse_failure} wall_s={elapsed:.1f}"
        )
        if args.max_episodes > 0 and episodes_run >= args.max_episodes:
            break

    env.close()
    wall = time.time() - t0
    print(
        f"Done. episodes_run={episodes_run} ok={episodes_ok} failed={episodes_failed} "
        f"total_env_steps={total_env_steps} total_llm_failures={total_llm_failures} wall_s={wall:.1f}\n"
        f"Logs under: {out_dir}/logs/"
    )


if __name__ == "__main__":
    main()


