"""Reproducible evaluation reports for PufferDrive policies.

Evaluates a checkpoint with the WOSAC realism metrics on a fixed set of scenes (an eval suite),
renders videos of a fixed subset of those scenes, and regenerates a static HTML dashboard that
compares every run evaluated on the suite.

The suite is created on first use and saved to <report_dir>/suites/<name>.json. It pins the scenes
and the evaluation protocol (init steps, control mode, rollouts, seed), so later runs evaluated on
the same suite are directly comparable.

Usage (from the repo root; accepts the usual puffer config overrides, e.g. --env.dynamics-model):
    python -m pufferlib.ocean.benchmark.eval_report --run-name self_play \\
        --load-model-path experiments/puffer_drive_179319117119.pt
    python -m pufferlib.ocean.benchmark.eval_report --rebuild-only  # only regenerate the dashboard

Then open <report_dir>/index.html in a browser.
"""

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

import pufferlib
import pufferlib.pytorch
from pufferlib.pufferl import load_config, load_env, load_policy
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

ENV_NAME = "puffer_drive"
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "report_template.html")

# Per-scene component likelihoods grouped as in the WOSAC leaderboard
METRIC_GROUPS = {
    "kinematic_metrics": [
        "likelihood_linear_speed",
        "likelihood_linear_acceleration",
        "likelihood_angular_speed",
        "likelihood_angular_acceleration",
    ],
    "interactive_metrics": [
        "likelihood_collision_indication",
        "likelihood_distance_to_nearest_object",
        "likelihood_time_to_collision",
    ],
    "map_based_metrics": [
        "likelihood_distance_to_road_edge",
        "likelihood_offroad_indication",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a policy on a fixed scene suite and build the dashboard")
    parser.add_argument(
        "--run-name", type=str, default=None, help="Name shown in the dashboard (default: checkpoint name)"
    )
    parser.add_argument("--notes", type=str, default="", help="Free-text description of the run")
    parser.add_argument("--suite", type=str, default="default", help="Eval suite name (created on first use)")
    parser.add_argument("--report-dir", type=str, default="eval_reports")
    parser.add_argument("--num-scenes", type=int, default=64, help="Scenes in a new suite")
    parser.add_argument("--num-videos", type=int, default=8, help="Scenes rendered to video in a new suite")
    parser.add_argument("--seed", type=int, default=0, help="Scene sampling and rollout seed of a new suite")
    parser.add_argument("--no-videos", action="store_true", help="Skip rendering videos")
    parser.add_argument("--rebuild-only", action="store_true", help="Only regenerate index.html")
    report_args, remaining = parser.parse_known_args()
    # Leave the rest of the command line to the regular puffer config parser
    sys.argv = [sys.argv[0]] + remaining
    return report_args


def protocol_from_config(args):
    eval_cfg = args["eval"]
    return dict(
        init_steps=int(eval_cfg["wosac_init_steps"]),
        control_mode=eval_cfg["wosac_control_mode"],
        init_mode=eval_cfg["wosac_init_mode"],
        goal_behavior=int(eval_cfg["wosac_goal_behavior"]),
        goal_radius=float(eval_cfg["wosac_goal_radius"]),
        num_rollouts=int(eval_cfg["wosac_num_rollouts"]),
        batch_size=int(eval_cfg["wosac_batch_size"]),
    )


def apply_protocol(args, map_dir, protocol, map_ids, render_mode=None):
    """Configure args to build an eval env over exactly `map_ids` under the suite protocol."""
    env = args["env"]
    env["map_dir"] = map_dir
    env["map_ids"] = [int(m) for m in map_ids]
    env["init_mode"] = protocol["init_mode"]
    env["control_mode"] = protocol["control_mode"]
    env["init_steps"] = protocol["init_steps"]
    env["goal_behavior"] = protocol["goal_behavior"]
    env["goal_radius"] = protocol["goal_radius"]
    env["episode_length"] = 91
    env["termination_mode"] = 0
    env["resample_frequency"] = 0
    if render_mode is not None:
        env["render_mode"] = render_mode
    args["vec"] = dict(backend="PufferEnv", num_envs=1)
    args["eval"]["wosac_init_steps"] = protocol["init_steps"]
    args["eval"]["wosac_num_rollouts"] = protocol["num_rollouts"]


def available_map_ids(map_dir):
    ids = []
    for path in glob.glob(os.path.join(map_dir, "map_*.bin")):
        match = re.fullmatch(r"map_(\d+)\.bin", os.path.basename(path))
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def create_suite(args, report_args, suite_path):
    """Sample scenes with controllable agents and at least one vehicle track to predict, and save the suite.

    Scenes without a vehicle track to predict have no time-to-collision likelihood (it is computed for
    vehicles only) and would be dropped by compute_metrics.
    """
    map_dir = args["eval"]["map_dir"]
    protocol = protocol_from_config(args)
    rng = np.random.default_rng(report_args.seed)
    candidates = list(rng.permutation(available_map_ids(map_dir)))
    if not candidates:
        raise pufferlib.APIUsageError(f"No map_*.bin files found in {map_dir}")

    print(f"Creating eval suite '{report_args.suite}' with {report_args.num_scenes} scenes from {map_dir}")
    scenes, seen = [], set()
    while len(scenes) < report_args.num_scenes and candidates:
        chunk, candidates = candidates[: report_args.num_scenes], candidates[report_args.num_scenes :]
        apply_protocol(args, map_dir, protocol, chunk)
        vecenv = load_env(ENV_NAME, args)
        gt = vecenv.get_ground_truth_trajectories()
        scenes_with_tracks = set(gt["scenario_id"][gt["is_track_to_predict"] & gt["is_vehicle"]].tolist())
        for map_id, scenario_id in zip(vecenv.map_ids, vecenv.scenario_ids):
            if scenario_id in scenes_with_tracks and scenario_id not in seen and len(scenes) < report_args.num_scenes:
                scenes.append(dict(map_id=int(map_id), scenario_id=scenario_id))
                seen.add(scenario_id)
        vecenv.close()

    if len(scenes) < report_args.num_scenes:
        print(f"Warning: only found {len(scenes)} usable scenes in {map_dir}")

    suite = dict(
        name=report_args.suite,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        map_dir=map_dir,
        seed=report_args.seed,
        protocol=protocol,
        scenes=scenes,
        video_scenes=[s["scenario_id"] for s in scenes[: report_args.num_videos]],
    )
    os.makedirs(os.path.dirname(suite_path), exist_ok=True)
    with open(suite_path, "w") as f:
        json.dump(suite, f, indent=2)
    print(f"Saved eval suite to {suite_path}")
    return suite


def rollout(args, vecenv, policy, num_rollouts, num_steps, render=False):
    """Roll out the policy from the env's initial state.

    Returns:
        trajectories: dict with keys 'x', 'y', 'z', 'heading', 'id', each (num_agents, num_rollouts, num_steps)
        env_logs: list with one dict per env, averaged over all agents and rollouts
    """
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args["train"]["device"]

    trajectories = {
        key: np.zeros((num_agents, num_rollouts, num_steps), dtype=np.int32 if key == "id" else np.float32)
        for key in ["x", "y", "z", "heading", "id"]
    }

    for rollout_idx in range(num_rollouts):
        obs, _ = vecenv.reset()
        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        for time_idx in range(num_steps):
            agent_state = driver.get_global_agent_state()
            for key in trajectories:
                trajectories[key][:, rollout_idx, time_idx] = agent_state[key]

            if render:
                driver.render(env_id=0)

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, _ = policy.forward_eval(ob_tensor, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)

            # per_env_logs keeps vec_log from consuming the logs, so they accumulate over all rollouts
            obs, _, _, _, _ = vecenv.step(action_np, per_env_logs=True)

    return trajectories, driver.get_env_logs()


def scene_level_metrics(df):
    """Recompute the metric groups per scene (compute_metrics fills them with the batch mean)."""
    df = df.copy()
    for group, columns in METRIC_GROUPS.items():
        df[group] = df[columns].mean(axis=1)
    return df


def env_log_metrics(log):
    """Task metrics from the env's own episode logs (all controlled agents, not just tracks to predict).

    goals_reached_this_episode is not usable as a rate here: with goal_behavior=2 (stop) it counts every
    step an agent spends at its goal. score (goal reached without colliding) is unaffected.
    """
    if not log:
        return {}
    return {
        "env_score": log["score"],
        "env_collision_rate": log["collision_rate"],
        "env_offroad_rate": log["offroad_rate"],
    }


def evaluate_suite(args, suite):
    """WOSAC metrics and env task metrics for every scene in the suite, in batches of protocol['batch_size']."""
    protocol, map_dir = suite["protocol"], suite["map_dir"]
    scenes = suite["scenes"]
    evaluator = WOSACEvaluator(args)
    policy = None
    frames, env_metrics = [], {}

    batch_size = protocol["batch_size"]
    for start in range(0, len(scenes), batch_size):
        batch = scenes[start : start + batch_size]
        print(f"Evaluating scenes {start + 1}-{start + len(batch)} of {len(scenes)}")
        apply_protocol(args, map_dir, protocol, [s["map_id"] for s in batch])
        vecenv = load_env(ENV_NAME, args)
        if vecenv.scenario_ids != [s["scenario_id"] for s in batch]:
            raise RuntimeError(
                f"Maps in {map_dir} no longer match suite '{suite['name']}' (expected scenario ids "
                f"{[s['scenario_id'] for s in batch]}, got {vecenv.scenario_ids})"
            )
        if policy is None:
            policy = load_policy(args, vecenv, ENV_NAME)
            policy.eval()

        gt = evaluator.collect_ground_truth_trajectories(vecenv)
        sim, env_logs = rollout(args, vecenv, policy, protocol["num_rollouts"], evaluator.sim_steps)
        frames.append(
            evaluator.compute_metrics(
                gt,
                sim,
                vecenv.get_global_agent_state(),
                vecenv.get_road_edge_polylines(),
                aggregate_results=False,
                drop_last_scenario=False,
            )
        )
        for scenario_id, log in zip(vecenv.scenario_ids, env_logs):
            env_metrics[scenario_id] = env_log_metrics(log)
        vecenv.close()

    df = scene_level_metrics(pd.concat(frames))
    df = df.join(pd.DataFrame.from_dict(env_metrics, orient="index"))
    missing = [s["scenario_id"] for s in scenes if s["scenario_id"] not in df.index]
    if missing:
        print(f"Warning: no WOSAC metrics for {len(missing)} suite scene(s), excluded from the means: {missing}")
    return df, policy


def render_videos(args, suite, policy, video_dir):
    """Render one rollout per video scene to <video_dir>/<scenario_id>.mp4."""
    protocol, map_dir = suite["protocol"], suite["map_dir"]
    by_id = {s["scenario_id"]: s for s in suite["scenes"]}
    num_steps = WOSACEvaluator(args).sim_steps
    os.makedirs(video_dir, exist_ok=True)
    videos = {}

    for i, scenario_id in enumerate(suite["video_scenes"]):
        print(f"Rendering scene {i + 1}/{len(suite['video_scenes'])}: {scenario_id}")
        apply_protocol(args, map_dir, protocol, [by_id[scenario_id]["map_id"]], render_mode=1)
        vecenv = load_env(ENV_NAME, args)
        started = time.time()
        rollout(args, vecenv, policy, num_rollouts=1, num_steps=num_steps, render=True)
        vecenv.close()  # Flushes the ffmpeg pipe

        # The headless renderer writes <scenario_id>.mp4 to the working directory
        written = [p for p in glob.glob("*.mp4") if os.path.getmtime(p) >= started - 1]
        if len(written) != 1:
            print(f"Warning: expected one rendered video for {scenario_id}, found {written}")
            continue
        shutil.move(written[0], os.path.join(video_dir, f"{scenario_id}.mp4"))
        videos[scenario_id] = f"videos/{scenario_id}.mp4"

    return videos


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True))
        return dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return {}


def to_builtin(value):
    """JSON-safe scalar: numpy types to Python, NaN to None."""
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_dashboard(report_dir):
    suites = {}
    for path in sorted(glob.glob(os.path.join(report_dir, "suites", "*.json"))):
        with open(path) as f:
            suite = json.load(f)
        suites[suite["name"]] = suite

    runs = []
    for path in glob.glob(os.path.join(report_dir, "runs", "*", "metrics.json")):
        with open(path) as f:
            run = json.load(f)
        run["dir"] = os.path.relpath(os.path.dirname(path), report_dir)
        runs.append(run)
    runs.sort(key=lambda r: r["created"])

    with open(TEMPLATE_PATH) as f:
        template = f.read()
    # Escape "</" so the JSON cannot close the <script> tag it is embedded in
    data = json.dumps(dict(suites=suites, runs=runs)).replace("</", "<\\/")
    html = template.replace("/*__REPORT_DATA__*/ null", data)

    index_path = os.path.join(report_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(html)
    print(f"Dashboard with {len(runs)} run(s): {os.path.abspath(index_path)}")


def main():
    report_args = parse_args()
    if report_args.rebuild_only:
        build_dashboard(report_args.report_dir)
        return

    args = load_config(ENV_NAME)
    checkpoint = args["load_model_path"]
    if checkpoint is None and args["load_id"] is None:
        raise pufferlib.APIUsageError("Pass --load-model-path (or --load-id) with the checkpoint to evaluate")
    if checkpoint == "latest":
        checkpoint = max(glob.glob(f"experiments/{ENV_NAME}*.pt"), key=os.path.getctime)
        args["load_model_path"] = checkpoint

    run_name = report_args.run_name or os.path.splitext(os.path.basename(checkpoint or args["load_id"]))[0]
    run_dir_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
    run_dir = os.path.join(report_args.report_dir, "runs", run_dir_name)

    suite_path = os.path.join(report_args.report_dir, "suites", f"{report_args.suite}.json")
    if os.path.exists(suite_path):
        with open(suite_path) as f:
            suite = json.load(f)
        print(f"Using eval suite '{suite['name']}': {len(suite['scenes'])} scenes, protocol {suite['protocol']}")
    else:
        suite = create_suite(args, report_args, suite_path)

    torch.manual_seed(suite["seed"])
    np.random.seed(suite["seed"])

    df, policy = evaluate_suite(args, suite)

    if os.path.exists(run_dir):
        print(f"Overwriting existing run '{run_name}' in {run_dir}")
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    videos = {}
    if not report_args.no_videos and suite["video_scenes"]:
        videos = render_videos(args, suite, policy, os.path.join(run_dir, "videos"))

    aggregate = {k: to_builtin(v) for k, v in df.mean().items()}
    aggregate["realism_meta_score_std"] = to_builtin(df["realism_meta_score"].std())
    aggregate["num_scenes"] = len(df)
    aggregate["num_agents"] = int(df["num_agents_per_scene"].sum())

    env_config = {k: to_builtin(v) for k, v in args["env"].items() if k != "map_ids"}
    run = dict(
        run_name=run_name,
        notes=report_args.notes,
        suite=suite["name"],
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        checkpoint=checkpoint or f"load_id:{args['load_id']}",
        git=git_info(),
        protocol=suite["protocol"],
        env_config=env_config,
        aggregate=aggregate,
        scenes={sid: {k: to_builtin(v) for k, v in row.items()} for sid, row in df.iterrows()},
        videos=videos,
    )
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(run, f, indent=2)

    print(f"\nRealism meta score: {aggregate['realism_meta_score']:.4f} over {aggregate['num_scenes']} scenes")
    build_dashboard(report_args.report_dir)


if __name__ == "__main__":
    main()
