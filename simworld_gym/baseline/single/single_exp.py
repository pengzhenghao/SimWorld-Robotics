import os
print("Start script")
import cv2
import gym, simworld_gym
import numpy as np
import argparse
import warnings
from typing import List
import traceback
from pathlib import Path

from baseline_utils import numpy_to_base64, split_into_strips, action_history_text, save_images, log_lines
from agents import ReasoningAgent, ReActAgent
from prompt_template import nav_template, reasoning_template, perception_template
from exp_artifacts import EpisodeLogger, render_mp4_from_frames, utc_timestamp, write_json

warnings.filterwarnings("ignore")

# numpy 2.x removed bool8 alias; Gym still references np.bool8
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


def save_video(frames: List, video_path: str, fps: int = 10):
    """Save a list of RGB frames (HWC uint8) to mp4."""
    if not frames:
        print("No frames to save for video.")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for frame in frames:
        # OpenCV's VideoWriter expects BGR; Gym/SimWorld provides RGB.
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved video to {video_path}")

parser = argparse.ArgumentParser(description="Run the agent in a simulated environment.")
parser.add_argument("--map", type=str, default="20_0", help="Number of the World.")
parser.add_argument("--backend", type=str, default="openai", help="Backend for the LLM.")
parser.add_argument("--model", type=str, default="gpt-4o", help="Model name.")
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--reasoning", action="store_true", help="Use ReasoningAgent (single-step plan) instead of ReAct.")
parser.add_argument("--setting", type=str, choices=["simple", "traffic"], default="simple", help="Task setting: simple or traffic.")
parser.add_argument("--strip", action="store_true", help="Use strip mode for perception.")
parser.add_argument("--depth", action="store_true", help="Use depth mode for perception.")
parser.add_argument("--segment", action="store_true", help="Use segmentation mode for perception.")

args = parser.parse_args()
map = args.map
backend = args.backend
ip = args.ip
setting = args.setting
model = args.model
reasoning = args.reasoning
log_dir = f"{model}_{setting}"
strip = args.strip
depth = args.depth
segment = args.segment

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

ue_port = int(os.getenv("UE_PORT", "9000"))

print(f"Map: {map}, Backend: {backend}, Model: {model}, UE Port: {ue_port}")

reward_setting = {
    "human_collision_penalty": -1,
    "object_collision_penalty": -0.1,
    "action_penalty": -0.1,
    "success_reward": 10.0,
    "off_track_penalty": -0.01
}
env = gym.make(
    'simworld_gym/SimpleWorld' if setting == "simple" else 'simworld_gym/TrafficWorld',
    port=ue_port,
    resolution=(720, 600),
    render_mode="rgb_array",
    observation_type="all",
    record_video=False,
    log_dir=log_dir,
    reward_setting=reward_setting,
)

# Initialize agent based on mode
if reasoning:
    agent = ReasoningAgent(
        backend=backend,
        model=model,
        system_prompt=nav_template(strip=strip, depth=depth, segment=segment),
    )
else:
    agent = ReActAgent(
        backend=backend,
        model=model,
        reasoning_prompt=reasoning_template(strip=strip),
        perception_prompt=perception_template(strip=strip, depth=depth, segment=segment),
    )

initialized = False

action_mapping = ["Move_Forward", "Rotate_Left", "Rotate_Right", "Move_Left", "Move_Right", "Subtask_Completed"]
task = 11 if int(map.split("_")[1]) < 50 else 21
task_2_test = [f"task_dist_{task}_0_1"]

input_tokens = 0
output_tokens = 0

# Resolve writable output root
output_root = os.environ.get("SIMWORLD_OUTPUT_DIR", os.path.join(os.getcwd(), "agent_log"))
output_root = os.path.abspath(output_root)
os.makedirs(output_root, exist_ok=True)
print(f"Logging to: {output_root}")

# Create a per-run directory so artifacts don't collide across invocations.
run_id = os.environ.get("SIMWORLD_RUN_ID", utc_timestamp())
run_dir = os.path.join(output_root, log_dir, run_id)
os.makedirs(run_dir, exist_ok=True)
write_json(
    os.path.join(run_dir, "run_config.json"),
    {
        "backend": backend,
        "model": model,
        "setting": setting,
        "map": map,
        "reasoning": bool(reasoning),
        "strip": bool(strip),
        "depth": bool(depth),
        "segment": bool(segment),
        "ue_port": ue_port,
        "run_id": run_id,
    },
)

for task in task_2_test:
    base_task_dir = os.path.join("single_agent_world", "easy", f"map_road_{map}")
    # Fallback to bundled simple data if "easy" split is not available locally
    if not os.path.exists(os.path.join(base_task_dir, task)):
        sample_task_dir = os.path.join("single_agent_world", "simple", f"map_road_{map}")
        if os.path.exists(os.path.join(sample_task_dir, task)):
            print(f'WARNING: task "{task}" not found under "easy". Using simple data instead.')
            base_task_dir = sample_task_dir
    task_path = os.path.join(base_task_dir, task)
    print(f"Using task path: {task_path}")
    world_json = os.path.join(task_path, "progen_world.json")
    agent_json = os.path.join(task_path, "task_config.json")
    if not initialized:
        options = {
            "task_path": task_path,
            "agent_json": agent_json,
            "world_json": world_json,
        }
        if setting != "simple":
            traffic_json = os.path.join(task_path, "traffic.json")
            options["traffic_json"] = traffic_json
        initialized = True
    else:
        options = {
            "task_path": task_path,
            "agent_json": agent_json,
        }
        if setting != "simple":
            traffic_json = os.path.join(task_path, "traffic.json")
            options["traffic_json"] = traffic_json
    observation, info = env.reset(options=options)
    vision_cue = info["current_instruction"]["image"]
    instruction = info["current_instruction"]["text"]
    action_history = []
    chosen_actions = []
    frames = []
    
    # Reset agent state for new task
    agent.reset_state()
    
    folder_path = os.path.join(run_dir, f"{map}", f"{task}")
    os.makedirs(folder_path , exist_ok=True)
    print(f"Saving run artifacts under: {folder_path}")
    ep_logger = EpisodeLogger.for_episode(folder_path)

    i = 0
    terminated = False
    last_position = None
    current_position = None
    parse_failure_count = 0
    ACTION_TO_STEP = {0: 0, 1: 5, 2: 4, 3: 2, 4: 3}

    while True:
        orientation = info['agent']['agent_rotation']
        current_position = info['agent']['agent_location']
        forward_count = sum(1 for a in chosen_actions if a == 0)
        if not last_position is None and forward_count > 3 and np.linalg.norm(np.array(current_position) - np.array(last_position)) < 0.5:
            print("Stuck in place, terminating.")
            log_lines(folder_path, [("STUCK", "")])
            break
        last_position = current_position
        frames.append(observation["rgb"].copy())
        # Save decision-time images for debugging
        try:
            img_dir = os.path.join(folder_path, "agent_images")
            os.makedirs(img_dir, exist_ok=True)
            tag = f"{i:06d}"
            cv2.imwrite(os.path.join(img_dir, f"current_{tag}.png"), observation["rgb"])
            cv2.imwrite(os.path.join(img_dir, f"expected_{tag}.png"), vision_cue)
            save_images(observation["rgb"], vision_cue, os.path.join(img_dir, f"display_{tag}.png"))
        except Exception:
            pass

        # Preprocess images
        images = []
        if strip:
            strips = split_into_strips(observation["rgb"])
            for idx, img in enumerate(strips):
                images.append({"img": numpy_to_base64(img), "description": [
                    'The view on the left',
                    'The horizontal center',
                    'The right',
                ][idx] if idx < 3 else None})
            images.append({"img": numpy_to_base64(vision_cue), "description": 'The expected view'})
        else:
            images.append({"img": numpy_to_base64(observation["rgb"]), "description": 'The current view'})
            images.append({"img": numpy_to_base64(vision_cue), "description": 'The expected view'})
        if segment:
            images.append({"img": numpy_to_base64(observation["object_mask"]), "description": 'The object segmentation mask of the current view'})
        if depth:
            images.append({"img": numpy_to_base64(observation["depth_map"]), "description": 'The depth map of the current view'})
        
        # Preprocess action history text
        action_history_str = action_history_text(action_history, action_mapping)
        
        try:
            result = agent.step(
                observation=images,
                instruction=instruction,
                orientation=orientation,
                action_history_text=action_history_str,
                chosen_actions=chosen_actions,
            )
            chosen_actions = result.get("actions", None)
            vision_description = result.get("vision_description", "")
            summary = result.get("summary", "")
            match = result.get("match", None)
            reason = result.get("reason", "")
            usage = result.get("usage", {"input": 0, "output": 0})
            input_tokens += usage.get("input", 0)
            output_tokens += usage.get("output", 0)
            print(f"[vision {i}]", vision_description)
            if reason:
                print(f"[reason {i}]", reason)
            ep_logger.log_step(
                {
                    "event": "llm_step",
                    "step": i,
                    "instruction": instruction,
                    "orientation": orientation,
                    "agent_location": current_position,
                    "action_history": list(action_history),
                    "chosen_actions": list(chosen_actions) if chosen_actions else [],
                    "vision_description": vision_description,
                    "summary": summary,
                    "match": match,
                    "reason": reason,
                    "usage": usage,
                }
            )
            log_entries = [
                ("current subtask", instruction),
                (f"vision {i}", str(vision_description).replace("\n", "")),
                (f"summary {i}", str(summary).replace("\n", "")),
                (f"actions {i}", str(chosen_actions)),
            ]
            if reason:
                log_entries.insert(2, (f"reason {i}", str(reason).replace("\n", "")))
            if match is not None:
                log_entries.append((f"match {i}", str(match)))
            log_lines(folder_path, log_entries)
        except Exception as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            err_msg = (
                f"Agent step error: {e} | backend={backend} model={model} map={map} "
                f"task={task} step={i}"
            )
            print(err_msg)
            ep_logger.log_step(
                {
                    "event": "llm_error",
                    "step": i,
                    "instruction": instruction,
                    "orientation": orientation,
                    "agent_location": current_position,
                    "action_history": list(action_history),
                    "chosen_actions": list(chosen_actions) if chosen_actions else [],
                    "error": err_msg,
                    "traceback": tb_str,
                }
            )
            log_lines(folder_path, [
                ("ERROR", err_msg),
                ("TRACEBACK", tb_str),
            ])
            parse_failure_count += 1
            if parse_failure_count > 10:
                print("Too many errors, terminating.")
                log_lines(folder_path, [("TOO MANY ERRORS", "")])
                break
            continue
        if not chosen_actions:
            print("No action specified. Failing.")
            break
        for chosen_action in chosen_actions:
            action_history.append(chosen_action)
            i += 1
            if chosen_action == -1:
                observation, _, terminated, _, info = env.step(-1)
                frames.append(observation["rgb"].copy())
                if terminated:
                    print("End")
                    break
                instruction = info["current_instruction"]["text"]
                vision_cue = info["current_instruction"]["image"]
                action_history = []
            else:
                step_code = ACTION_TO_STEP.get(chosen_action)
                if step_code is None:
                    continue
                observation, _, terminated, _, info = env.step(step_code)
                frames.append(observation["rgb"].copy())
        if terminated:
            print("End")
            break

    # Save rollout video per task
    video_path = os.path.join(folder_path, "rollout.mp4")
    save_video(frames, video_path, fps=10)
    # Also render a quick observation video from saved decision-time images if present.
    try:
        img_dir = Path(folder_path) / "agent_images"
        paths = sorted(img_dir.glob("current_*.png"))
        if paths:
            out = render_mp4_from_frames(paths, Path(folder_path) / "observations.mp4", fps=10)
            if out:
                ep_logger.log_step({"event": "videos_rendered", "observations_mp4": str(out)})
    except Exception:
        pass
                    
env.close()