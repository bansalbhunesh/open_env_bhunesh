# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the support operations environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SupportOpsAction, SupportOpsObservation
    from ..evaluation import grade_episode_export, run_in_process_baseline
    from ..models import (
        BaselineResponse,
        GraderRequest,
        GraderResponse,
        TaskCatalogResponse,
    )
    from ..task_catalog import get_scenario, list_task_descriptors
    from .support_ops_env_environment import SupportOpsEnvironment
except ImportError:
    from models import SupportOpsAction, SupportOpsObservation
    from evaluation import grade_episode_export, run_in_process_baseline
    from models import (
        BaselineResponse,
        GraderRequest,
        GraderResponse,
        TaskCatalogResponse,
    )
    from task_catalog import get_scenario, list_task_descriptors
    from server.support_ops_env_environment import SupportOpsEnvironment


# Create the app with web interface and README integration
app = create_app(
    SupportOpsEnvironment,
    SupportOpsAction,
    SupportOpsObservation,
    env_name="support_ops_env",
    max_concurrent_envs=4,
)


@app.get("/tasks", response_model=TaskCatalogResponse, tags=["Environment Info"])
def list_tasks() -> TaskCatalogResponse:
    return TaskCatalogResponse(tasks=list_task_descriptors())


@app.post("/grader", response_model=GraderResponse, tags=["Environment Info"])
def grade_episode(request: GraderRequest) -> GraderResponse:
    scenario = get_scenario(request.task_id or request.episode.task_id)
    return grade_episode_export(request.episode, scenario)


@app.get("/baseline", response_model=BaselineResponse, tags=["Environment Info"])
@app.post("/baseline", response_model=BaselineResponse, tags=["Environment Info"])
def run_baseline() -> BaselineResponse:
    return run_in_process_baseline(SupportOpsEnvironment)


def serve(host: str = "0.0.0.0", port: int = 8000):
    """Run the uvicorn server with explicit host and port parameters."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main():
    """CLI entry point for `uv run server` and `python -m ...`."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
