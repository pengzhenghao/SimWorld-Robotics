from __future__ import annotations

import argparse
import os
import time
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import gym
import numpy as np
import simworld_gym

from agents import ReasoningAgent, ReActAgent
from baseline_utils import action_history_text, numpy_to_base64, save_images, split_into_strips
from exp_artifacts import EpisodeLogger, render_episode_videos, write_json
from prompt_template import nav_template, perception_template, reasoning_template


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_s(dt_s: float) -> str:
    if dt_s < 1:
        return f"{dt_s*1000:.0f}ms"
    return f"{dt_s:.2f}s"


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
        "--phase_warn_s",
        type=float,
        default=float(os.environ.get("SWEEP_PHASE_WARN_S", "10")),
        help="Print an extra warning line if a single phase takes longer than this many seconds. Default: 10.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="If set, SimWorldGym will save per-step images/videos under each episode's images/ directory (can be large).",
    )
    parser.add_argument(
        "--save_step_images",
        action="store_true",
        default=True,
        help="Save decision-time images (current + expected + side-by-side) under each episode dir for debugging.",
    )
    parser.add_argument(
        "--no_render_videos",
        action="store_true",
        help="Disable auto-rendering mp4 videos at the end of each episode (when record_video is enabled).",
    )
    parser.add_argument(
        "--eval_at_end",
        action="store_true",
        default=bool(int(os.environ.get("SWEEP_EVAL_AT_END", "0"))),
        help="If set, run evaluate.py at the end of the sweep and save the summary under the run log directory.",
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
    uw = env.unwrapped if hasattr(env, "unwrapped") else env

    # Write run metadata alongside env logs for reproducibility.
    try:
        run_meta = {
            "backend": args.backend,
            "model": args.model,
            "env": args.env,
            "split": args.split,
            "tasks": args.tasks,
            "maps": args.maps,
            "max_maps": args.max_maps,
            "ue_port": args.ue_port,
            "out_dir": out_dir,
            "strip": bool(args.strip),
            "depth": bool(args.depth),
            "segment": bool(args.segment),
            "reasoning": bool(args.reasoning),
            "record_video": bool(args.record_video),
            "save_step_images": bool(args.save_step_images),
        }
        base_dir = getattr(uw, "base_dir", None)
        if base_dir:
            write_json(os.path.join(base_dir, "run_config.json"), run_meta)
    except Exception:
        pass

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

        print(f"[{_now_str()}] [episode_start {idx}/{len(pairs)}] map={map_dir} task={task_dir}")
        agent.reset_state()
        t_reset0 = time.perf_counter()
        observation, info = env.reset(options=options)
        t_reset = time.perf_counter() - t_reset0
        if t_reset > args.phase_warn_s:
            print(f"[{_now_str()}] [PHASE slow] env.reset took {_fmt_s(t_reset)} map={map_dir} task={task_dir}")
        else:
            print(f"[{_now_str()}] [PHASE] env.reset {_fmt_s(t_reset)} map={map_dir} task={task_dir}")
        # Episode directory created by the env (contains trajectory.csv and images/).
        episode_dir = getattr(uw, "current_episode_dir", None)
        if episode_dir:
            print(f"[{_now_str()}] [episode_dir] {episode_dir}")
        ep_logger = EpisodeLogger.for_episode(episode_dir) if episode_dir else None
        if ep_logger:
            ep_logger.write_run_meta(
                {
                    "map": map_dir,
                    "task": task_dir,
                    "task_path": task_path,
                    "backend": args.backend,
                    "model": args.model,
                    "env": args.env,
                    "split": args.split,
                    "strip": bool(args.strip),
                    "depth": bool(args.depth),
                    "segment": bool(args.segment),
                    "reasoning": bool(args.reasoning),
                    "record_video": bool(args.record_video),
                },
                filename="episode_config.json",
            )

        vision_cue = info["current_instruction"]["image"]
        instruction = info["current_instruction"]["text"]
        chosen_actions: List[int] = []
        action_history: List[int] = []

        terminated = False
        parse_failure = 0
        step_i = 0
        ep_start = time.time()

        while not terminated:
            t_loop0 = time.perf_counter()
            orientation = info["agent"]["agent_rotation"]

            # Build model multimodal inputs: current view + expected view (+ optional aux)
            t_inputs0 = time.perf_counter()
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
            t_inputs = time.perf_counter() - t_inputs0
            if t_inputs > args.phase_warn_s:
                print(f"[{_now_str()}] [PHASE slow] build_inputs took {_fmt_s(t_inputs)} step={step_i} map={map_dir} task={task_dir}")

            action_history_str = action_history_text(action_history, action_mapping)

            # Optional: dump decision-time images for debugging.
            # - current view: observation["rgb"]
            # - expected view: vision_cue
            if ep_logger and (args.save_step_images or args.record_video):
                try:
                    step_dir = os.path.join(ep_logger.episode_dir, "agent_images")
                    os.makedirs(step_dir, exist_ok=True)
                    step_tag = f"{step_i:06d}"
                    cur_path = os.path.join(step_dir, f"current_{step_tag}.png")
                    exp_path = os.path.join(step_dir, f"expected_{step_tag}.png")
                    combo_path = os.path.join(step_dir, f"display_{step_tag}.png")
                    cv2.imwrite(cur_path, observation["rgb"])
                    cv2.imwrite(exp_path, vision_cue)
                    save_images(observation["rgb"], vision_cue, combo_path)
                except Exception:
                    pass

            try:
                print(
                    f"[{_now_str()}] [PHASE] agent.step start step={step_i} map={map_dir} task={task_dir} "
                    f"imgs={len(images)} hist_len={len(action_history)}"
                )
                t_llm0 = time.perf_counter()
                result = agent.step(
                    observation=images,
                    instruction=instruction,
                    orientation=orientation,
                    action_history_text=action_history_str,
                    chosen_actions=chosen_actions,
                )
                t_llm = time.perf_counter() - t_llm0
                chosen_actions = result.get("actions") or []
                print(
                    f"[{_now_str()}] [PHASE] agent.step done {_fmt_s(t_llm)} step={step_i} "
                    f"actions={chosen_actions}"
                )
                if t_llm > args.phase_warn_s:
                    print(f"[{_now_str()}] [PHASE slow] agent.step took {_fmt_s(t_llm)} step={step_i} map={map_dir} task={task_dir}")
                parse_failure = 0
                if ep_logger:
                    ep_logger.log_step(
                        {
                            "event": "llm_step",
                            "step": step_i,
                            "map": map_dir,
                            "task": task_dir,
                            "instruction": instruction,
                            "orientation": orientation,
                            "action_history": list(action_history),
                            "chosen_actions": list(chosen_actions),
                            "vision_description": result.get("vision_description"),
                            "summary": result.get("summary"),
                            "match": result.get("match"),
                            "reason": result.get("reason", ""),
                            "usage": result.get("usage", {}),
                            "images_meta": [x.get("description") for x in images if isinstance(x, dict)],
                        }
                    )
            except Exception as e:
                parse_failure += 1
                total_llm_failures += 1
                print(f"[warn] agent step failed (count={parse_failure}): {e}")
                if ep_logger:
                    ep_logger.log_step(
                        {
                            "event": "llm_error",
                            "step": step_i,
                            "map": map_dir,
                            "task": task_dir,
                            "instruction": instruction,
                            "orientation": orientation,
                            "action_history": list(action_history),
                            "chosen_actions": list(chosen_actions),
                            "error": str(e),
                        }
                    )
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
                print(
                    f"[{_now_str()}] [PHASE] env.step start step={step_i} model_action={a} "
                    f"mapped={ACTION_TO_STEP.get(a) if a != -1 else -1} map={map_dir} task={task_dir}"
                )
                t_env0 = time.perf_counter()
                if args.log_every_steps > 0 and (step_i % args.log_every_steps == 0):
                    cur_instr = (instruction or "").replace("\n", " ")
                    print(
                        f"[step] map={map_dir} task={task_dir} step={step_i} action={a} "
                        f"orientation={orientation} instr='{cur_instr[:120]}'"
                    )
                if a == -1:
                    observation, _, terminated, _, info = env.step(-1)
                    t_env = time.perf_counter() - t_env0
                    print(f"[{_now_str()}] [PHASE] env.step done {_fmt_s(t_env)} step={step_i} terminated={terminated}")
                    if t_env > args.phase_warn_s:
                        print(f"[{_now_str()}] [PHASE slow] env.step(-1) took {_fmt_s(t_env)} step={step_i} map={map_dir} task={task_dir}")
                    if ep_logger:
                        ep_logger.log_step(
                            {
                                "event": "env_step",
                                "step": step_i,
                                "env_action": -1,
                                "terminated": bool(terminated),
                                "info": info,
                            }
                        )
                    if terminated:
                        break
                    instruction = info["current_instruction"]["text"]
                    vision_cue = info["current_instruction"]["image"]
                    action_history = []
                else:
                    env_act = ACTION_TO_STEP.get(a)
                    if env_act is None:
                        print(f"[{_now_str()}] [warn] unknown model action={a} (skipping)")
                        continue
                    observation, _, terminated, _, info = env.step(env_act)
                    t_env = time.perf_counter() - t_env0
                    print(f"[{_now_str()}] [PHASE] env.step done {_fmt_s(t_env)} step={step_i} terminated={terminated}")
                    if t_env > args.phase_warn_s:
                        print(f"[{_now_str()}] [PHASE slow] env.step({env_act}) took {_fmt_s(t_env)} step={step_i} map={map_dir} task={task_dir}")
                    if ep_logger:
                        # When record_video is enabled, env saves frames using the *previous* steps_count
                        # and increments steps_count afterwards. So the frame index is steps_count - 1.
                        try:
                            frame_idx = int(getattr(uw, "steps_count", 0)) - 1
                            images_dir = getattr(uw, "images_dir", None)
                            obs_path = (
                                os.path.join(images_dir, f"observation_{frame_idx:06d}.png") if images_dir else None
                            )
                            act_dir = os.path.join(images_dir, f"action_{frame_idx:06d}") if images_dir else None
                        except Exception:
                            obs_path, act_dir = None, None
                        ep_logger.log_step(
                            {
                                "event": "env_step",
                                "step": step_i,
                                "env_action": env_act,
                                "terminated": bool(terminated),
                                "info": info,
                                "saved_observation_path": obs_path,
                                "saved_action_dir": act_dir,
                            }
                        )
                if terminated:
                    break
            t_loop = time.perf_counter() - t_loop0
            if t_loop > args.phase_warn_s:
                print(f"[{_now_str()}] [PHASE slow] step loop took {_fmt_s(t_loop)} step={step_i} map={map_dir} task={task_dir}")

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
        # Auto-render mp4 videos from env-saved frames if enabled.
        if args.record_video and not args.no_render_videos:
            try:
                episode_dir = getattr(uw, "current_episode_dir", None)
                fps = int(getattr(uw, "record_video_fps", 8) or 8)
                if episode_dir:
                    print(f"[{_now_str()}] [PHASE] render_videos start fps={fps} episode_dir={episode_dir}")
                    t_vid0 = time.perf_counter()
                    outs = render_episode_videos(episode_dir, fps=fps)
                    t_vid = time.perf_counter() - t_vid0
                    print(f"[{_now_str()}] [PHASE] render_videos done {_fmt_s(t_vid)} outputs={outs}")
                    if t_vid > args.phase_warn_s:
                        print(f"[{_now_str()}] [PHASE slow] render_videos took {_fmt_s(t_vid)} map={map_dir} task={task_dir}")
                    if ep_logger:
                        ep_logger.log_step({"event": "videos_rendered", "outputs": outs})
            except Exception as e:
                print(f"[warn] failed to render episode videos: {e}")
        if args.max_episodes > 0 and episodes_run >= args.max_episodes:
            break

    env.close()
    wall = time.time() - t0
    print(
        f"Done. episodes_run={episodes_run} ok={episodes_ok} failed={episodes_failed} "
        f"total_env_steps={total_env_steps} total_llm_failures={total_llm_failures} wall_s={wall:.1f}\n"
        f"Logs under: {out_dir}/logs/"
    )

    # Optional: summarize performance at the end of the run.
    if args.eval_at_end:
        try:
            eval_py = os.path.join(os.path.dirname(__file__), "evaluate.py")
            env_name = "simpleenv" if args.env == "simple" else "trafficenv"
            cmd = [
                sys.executable,
                eval_py,
                "--run_dir",
                out_dir,
                "--env",
                env_name,
                "--split",
                args.split,
            ]
            print(f"[{_now_str()}] [PHASE] evaluate start cmd={' '.join(cmd)}")
            t_eval0 = time.perf_counter()
            res = subprocess.run(cmd, capture_output=True, text=True)
            t_eval = time.perf_counter() - t_eval0
            print(f"[{_now_str()}] [PHASE] evaluate done {_fmt_s(t_eval)} exit={res.returncode}")
            if res.stdout:
                print(res.stdout.rstrip())
            if res.stderr:
                print(res.stderr.rstrip())
            # Save to file under the env base_dir if available, else under out_dir/logs.
            base_dir = getattr(uw, "base_dir", None)
            save_dir = base_dir if base_dir else os.path.join(out_dir, "logs")
            try:
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "evaluation.txt"), "w", encoding="utf-8") as f:
                    f.write(res.stdout or "")
                    if res.stderr:
                        f.write("\n--- stderr ---\n")
                        f.write(res.stderr)
            except Exception:
                pass
        except Exception as e:
            print(f"[warn] evaluate_at_end failed: {e}")


if __name__ == "__main__":
    main()


