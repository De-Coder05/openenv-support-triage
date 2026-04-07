from __future__ import annotations
from typing import Any

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from .models import SupportAction, SupportObservation, SupportState
except ImportError:
    from models import SupportAction, SupportObservation, SupportState
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class SupportEnvClient(EnvClient[SupportAction, SupportObservation, SupportState]):
    """
    Async client for the remote Support Triage environment.
    """

    def _step_payload(self, action: SupportAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "team": action.team,
            "question": action.question,
            "query_customer_id": action.query_customer_id,
            "resolution": action.resolution
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SupportObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportObservation(
            ticket_id=obs_data.get("ticket_id", ""),
            ticket_text=obs_data.get("ticket_text", ""),
            customer_id=obs_data.get("customer_id"),
            db_result=obs_data.get("db_result"),
            clarification_reply=obs_data.get("clarification_reply"),
            error=obs_data.get("error"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> SupportState:
        return SupportState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            expected_route=payload.get("expected_route"),
            customer_tier=payload.get("customer_tier"),
            clarification_asked=payload.get("clarification_asked", False),
            task_difficulty=payload.get("task_difficulty", "easy")
        )
