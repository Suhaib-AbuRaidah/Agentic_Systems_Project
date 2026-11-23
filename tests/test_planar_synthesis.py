from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness import WarmEnvironment, run

from articraft.agent.tools import ToolContext, get
from articraft.sdk._planar_synthesis import fourbar_example, slidercrank_example
from articraft.viewer import load_viewer_run

MAIN = """
import json
from pathlib import Path

from build123d import Box

from articraft.sdk import RigidBodyAssembly, TestContext
from articraft.sdk.planar import apply_planar_joints, planar_motion_states

result = json.loads(Path({result_path!r}).read_text())
object_model = RigidBodyAssembly("planar")
for item in result["mechanism"]["bodies"]:
    body = object_model.rigid_body(item["id"])
    body.add(Box(0.005, 0.005, 0.003), name="body")
layers = {{item["id"]: 0.006 * i for i, item in enumerate(result["mechanism"]["bodies"])}}
apply_planar_joints(object_model, result, layers=layers)
planar_targets = {targets}

def run_tests():
    ctx = TestContext(object_model)
    for state in planar_motion_states(object_model, result):
        with ctx.state(state):
            ctx.expect_no_collision("ground", "crank")
    return ctx.report()
"""


def task_spec(driver_only: bool) -> dict:
    spec = (slidercrank_example if driver_only else fourbar_example)(samples=7)
    spec["units"] = "m"
    for body in spec["bodies"]:
        body["pose"][:2] = [value / 1000 for value in body["pose"][:2]]
        for frame in body["frames"].values():
            frame[:2] = [value / 1000 for value in frame[:2]]
        for point in body.get("points", {}).values():
            point["xy"] = [value / 1000 for value in point["xy"]]
            point["bounds"] = {
                axis: [value / 1000 for value in limits]
                for axis, limits in point.get("bounds", {}).items()
            }
    if driver_only:
        del spec["targets"]
        spec["design"] = {"mode": "explicit"}
        next(joint for joint in spec["joints"] if joint["id"] == "rail")["limits"] = [0.08, 0.12]
    else:
        spec["targets"][0]["positions"] = [
            [x / 1000, y / 1000] for x, y in spec["targets"][0]["positions"]
        ]
    spec["options"].update(trajectory_tolerance=0.0005, closure_tolerance=1e-6)
    return spec


@pytest.mark.parametrize("driver_only", [False, True])
def test_planar_tool_and_compiler_verify_saved_linkage(tmp_path: Path, driver_only: bool):
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("planar")
    workspace = run_dir / "workspace"
    spec = task_spec(driver_only)
    task = workspace / "planar_task.json"
    task.write_text(json.dumps(spec))
    feedback = run(
        get("synthesize_planar_mechanism").run(
            ToolContext(env, run_dir, workspace), {"task_path": task.name}
        )
    )
    assert feedback["status"] == "success", feedback
    assert feedback["validation"]["continuation_pass"]
    report = json.loads((workspace / feedback["result_path"]).read_text())
    assert len(report["motion"]["intermediate"]) == 12
    targets = "[]"
    if not driver_only:
        assert feedback["design_changes"]
        tip = next(body for body in report["mechanism"]["bodies"] if body["id"] == "coupler")
        x, y = tip["points"]["trace"]["xy"]
        targets = f'[object_model.get_rigid_body("coupler").at(({x!r}, {y!r}, 0))]'
    main = workspace / "main.py"
    if not driver_only:
        main.write_text(MAIN.format(result_path=feedback["result_path"], targets="[]"))
        rejected = env.compile_path(run_dir)
        assert rejected["status"] == "error"
        assert "coupler.trace" in rejected["error"]
    main.write_text(MAIN.format(result_path=feedback["result_path"], targets=targets))
    result = env.compile_path(run_dir)
    assert result["status"] == "success", result.get("error") or result.get("test_report")
    assert "planar loop motion" in result["test_report"]["checks"]
    viewer = load_viewer_run(run_dir)
    version = Path(result["usdz"]).stem
    frames = json.loads(viewer.trajectories[version].read_text())["body_frames"]
    assert len(frames) == 19
    follower = "slider" if driver_only else "coupler"
    assert frames[0]["bodies"][follower]["pos"] != frames[-1]["bodies"][follower]["pos"]

    if not driver_only:
        main.write_text(main.read_text().replace(targets, targets.replace(repr(x), repr(x + 0.03))))
        rejected = env.compile_path(run_dir)
        assert rejected["status"] == "error"
        assert "planar waypoint position" in str(rejected["test_report"])
    task.write_text(task.read_text() + "\n")
    assert "Planar task changed" in env.compile_path(run_dir)["error"]
