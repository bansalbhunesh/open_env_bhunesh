# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the support operations environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SupportOpsAction, SupportOpsObservation, SupportOpsState


class SupportOpsEnv(
    EnvClient[SupportOpsAction, SupportOpsObservation, SupportOpsState]
):
    """Persistent WebSocket client for support task rollouts."""

    def _step_payload(self, action: SupportOpsAction) -> Dict:
        return action.model_dump(exclude_none=True, mode="json")

    def _parse_result(self, payload: Dict) -> StepResult[SupportOpsObservation]:
        observation = SupportOpsObservation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SupportOpsState:
        return SupportOpsState.model_validate(payload)
