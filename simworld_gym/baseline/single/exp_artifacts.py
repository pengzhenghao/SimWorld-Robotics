from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)


def append_jsonl(path: str | Path, obj: dict) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        # `info` dicts often contain numpy arrays (e.g. positions, images). Make dumping robust.
        try:
            f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
        except Exception as e:
            # Never crash the experiment due to logging.
            f.write(
                json.dumps(
                    {
                        "event": "logger_error",
                        "error": str(e),
                        "payload_type": str(type(obj)),
                        "payload_repr": repr(obj)[:2000],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def safe_imwrite(path: str | Path, img) -> bool:
    try:
        p = Path(path)
        ensure_dir(p.parent)
        return bool(cv2.imwrite(str(p), img))
    except Exception:
        return False


_INT_RE = re.compile(r"(\d+)")


def _json_default(o: Any):
    """
    Best-effort JSON serializer for common non-JSON types in this project.

    - numpy scalars -> Python scalars
    - numpy arrays -> small arrays tolist(), large arrays summarized (shape/dtype only)
    - Path -> str
    - bytes -> utf-8 (lossy) string
    """
    # Path-like
    if isinstance(o, Path):
        return str(o)

    # numpy scalars
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()

    # numpy arrays (avoid dumping huge image tensors into JSONL)
    if isinstance(o, np.ndarray):
        if o.size <= 64:
            return o.tolist()
        return {"__ndarray__": True, "shape": list(o.shape), "dtype": str(o.dtype)}

    # bytes
    if isinstance(o, (bytes, bytearray)):
        try:
            return o.decode("utf-8", errors="replace")
        except Exception:
            return repr(o)

    # Generic fallback
    return str(o)


def _extract_int(text: str) -> int:
    m = _INT_RE.search(text)
    return int(m.group(1)) if m else -1


def _sorted_paths(paths: Iterable[Path], key_fn) -> list[Path]:
    return sorted(list(paths), key=key_fn)


def render_mp4_from_frames(
    frame_paths: list[Path],
    out_mp4: Path,
    fps: int = 10,
    codec: str = "libx264",
) -> Optional[Path]:
    """
    Render an mp4 from a list of image files (best-effort).

    IMPORTANT:
    - OpenCV VideoWriter + mp4v can produce videos that look fine on Linux but show
      as green/garbled in QuickTime on macOS due to codec/pixel-format issues.
    - Prefer H.264 (libx264) with yuv420p via imageio-ffmpeg for broad compatibility.
    """
    if not frame_paths:
        return None

    ensure_dir(out_mp4.parent)

    # Preferred path: imageio-ffmpeg (bundled ffmpeg) + H.264 yuv420p
    try:
        import imageio.v2 as imageio  # type: ignore

        first = imageio.imread(frame_paths[0])
        h, w = first.shape[:2]

        with imageio.get_writer(
            out_mp4.as_posix(),
            fps=float(fps),
            codec=codec,
            # Ensure a QuickTime-friendly pixel format without duplicating -pix_fmt flags.
            ffmpeg_params=["-vf", "format=yuv420p"],
            macro_block_size=None,
            quality=8,
        ) as writer:
            for p in frame_paths:
                img = imageio.imread(p)
                if img is None:
                    continue
                if img.shape[:2] != (h, w):
                    # Resize with OpenCV (expects BGR), but our img is RGB -> keep it RGB by using PIL via cv2 fallback.
                    img = cv2.resize(img, (w, h))
                writer.append_data(img)
        return out_mp4
    except Exception:
        pass

    # Fallback: OpenCV VideoWriter (less compatible with macOS QuickTime).
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        return None
    try:
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
    finally:
        writer.release()
    return out_mp4


def render_episode_videos(
    episode_dir: str | Path,
    fps: int = 10,
    observation_glob: str = "observation_*.png",
    action_dir_glob: str = "action_*",
) -> dict[str, Optional[str]]:
    """
    Render videos for a SimWorldGym episode directory produced by record_video:
      - <episode_dir>/images/observation_*.png
      - <episode_dir>/images/action_XXXXXX/frame_*.png

    Writes:
      - <episode_dir>/videos/observations.mp4
      - <episode_dir>/videos/actions.mp4 (concatenated action frames)
    """
    episode_dir = Path(episode_dir)
    images_dir = episode_dir / "images"
    videos_dir = ensure_dir(episode_dir / "videos")

    obs_paths = _sorted_paths(
        images_dir.glob(observation_glob),
        key_fn=lambda p: _extract_int(p.stem),
    )
    obs_out = render_mp4_from_frames(obs_paths, videos_dir / "observations.mp4", fps=fps)

    # Concatenate action frames across all action dirs, keeping order by action index then frame index.
    action_frames: list[Path] = []
    action_dirs = _sorted_paths(
        [p for p in images_dir.glob(action_dir_glob) if p.is_dir()],
        key_fn=lambda p: _extract_int(p.name),
    )
    for d in action_dirs:
        frames = _sorted_paths(
            d.glob("frame_*.png"),
            key_fn=lambda p: _extract_int(p.stem),
        )
        action_frames.extend(frames)

    act_out = render_mp4_from_frames(action_frames, videos_dir / "actions.mp4", fps=fps)

    return {
        "observations_mp4": str(obs_out) if obs_out else None,
        "actions_mp4": str(act_out) if act_out else None,
    }


@dataclass
class EpisodeLogger:
    episode_dir: Path
    steps_jsonl: Path

    @classmethod
    def for_episode(cls, episode_dir: str | Path) -> "EpisodeLogger":
        episode_dir = Path(episode_dir)
        steps_jsonl = episode_dir / "agent_steps.jsonl"
        return cls(episode_dir=episode_dir, steps_jsonl=steps_jsonl)

    def log_step(self, payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        append_jsonl(self.steps_jsonl, payload)

    def write_run_meta(self, payload: dict, filename: str = "agent_run.json") -> None:
        write_json(self.episode_dir / filename, payload)


