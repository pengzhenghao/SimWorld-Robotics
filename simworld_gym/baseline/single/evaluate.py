from __future__ import annotations

import argparse
import os
import re
from glob import glob

import pandas as pd

from metric import (
    distance_progress,
    dynamic_collision,
    intersection_sr,
    ndtw,
    spl,
    static_collision,
    subtask_sr,
    violation,
)


def _parse_world_task_from_reset_dir(reset_dir_name: str) -> tuple[str, str]:
    """
    reset dir name pattern (created by env):
      reset_{N}_{world}_{task}
    where world is typically "map_road_20_0" and task is typically "task_dist_11_0_1".
    """
    m = re.match(r"^reset_\d+_(.+)$", reset_dir_name)
    if not m:
        raise ValueError(f"Unrecognized reset dir: {reset_dir_name}")
    rest = m.group(1)
    idx = rest.find("_task_dist_")
    if idx == -1:
        raise ValueError(f"Cannot find task marker in reset dir: {reset_dir_name}")
    world = rest[:idx]
    task = rest[idx + 1 :]  # drop leading "_"
    return world, task


def load_task_template(split: str, world: str, task: str) -> dict:
    # Use simworld_gym's own helper so paths match how env loads settings.
    from simworld_gym.utils import misc

    rel = os.path.join("single_agent_world", split, world, task, "task_config.json")
    return misc.load_env_setting(rel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        help=(
            "Directory used as env log_dir in gym.make (e.g. 'gemini-2.5-flash_simple'). "
            "Trajectories will be searched under <run_dir>/logs/<env>_*/reset_*/trajectory.csv."
        ),
    )
    parser.add_argument("--env", choices=["simpleenv", "trafficenv"], default="simpleenv")
    parser.add_argument("--split", choices=["easy", "simple", "sample"], default="easy")
    parser.add_argument("--no_collisions", dest="with_collisions", action="store_false")
    parser.add_argument("--no_violations", dest="with_violations", action="store_false")
    parser.add_argument("--no_ndtw", dest="with_ndtw", action="store_false")
    parser.add_argument("--no_intersection_sr", dest="with_intersection_sr", action="store_false")
    args = parser.parse_args()

    traj_glob = os.path.join(args.run_dir, "logs", f"{args.env}_*", "reset_*", "trajectory.csv")
    paths = sorted(glob(traj_glob))
    if not paths:
        raise SystemExit(f"No trajectories found: {traj_glob}")

    # Aggregate
    n = 0
    sr = 0.0
    spl_sum = 0.0
    subtask_sr_sum = 0.0
    dp_sum = 0.0
    sc_sum = 0.0
    dc_sum = 0.0
    vio_sum = 0.0
    ndtw_sum = 0.0
    isr_sum = 0.0

    for log_path in paths:
        reset_dir = os.path.basename(os.path.dirname(log_path))
        world, task = _parse_world_task_from_reset_dir(reset_dir)
        template = load_task_template(args.split, world, task)

        log_df = pd.read_csv(log_path)
        n += 1

        cur_spl = spl(log_df, template)
        cur_subtask_sr = subtask_sr(log_df, template)
        cur_dp = distance_progress(log_df, template)

        spl_sum += cur_spl
        sr += 1.0 if cur_spl > 0 else 0.0
        subtask_sr_sum += cur_subtask_sr
        dp_sum += cur_dp

        if args.with_collisions:
            sc_sum += static_collision(log_df, template)
            dc_sum += dynamic_collision(log_df, template)
        if args.with_violations:
            vio_sum += violation(log_df, template)
        if args.with_ndtw:
            ndtw_sum += ndtw(log_df, template)
        if args.with_intersection_sr:
            isr_sum += intersection_sr(log_df, template)

    # Report
    print(f"episodes={n}")
    print(f"SR={sr / n:.4f}")
    print(f"SPL={spl_sum / n:.4f}")
    print(f"Subtask_SR={subtask_sr_sum / n:.4f}")
    print(f"Distance_Progress={dp_sum / n:.4f}")
    if args.with_collisions:
        print(f"Static_Collisions={sc_sum / n:.4f}")
        print(f"Dynamic_Collisions={dc_sum / n:.4f}")
    if args.with_violations:
        print(f"RedLight_Violations={vio_sum / n:.4f}")
    if args.with_ndtw:
        print(f"nDTW={ndtw_sum / n:.4f}")
    if args.with_intersection_sr:
        print(f"Intersection_SR={isr_sum / n:.4f}")


if __name__ == "__main__":
    main()