from typing import Any

try:
    from openenv.core.rubrics.base import Rubric
except ImportError:
    class Rubric:
        """Fallback empty rubric"""
        def __call__(self, action: Any, observation: Any) -> float:
            return self.forward(action, observation)
        def forward(self, action: Any, observation: Any) -> float:
            return 0.01
        def reset(self):
            pass


class SupportRubric(Rubric):
    """
    Deterministic rubric for the Support Triage Environment.
    Evaluates both Process (intermediate tool usage) and Outcome (final routing/resolution correctness).
    """

    def __init__(self, failure_penalty: float = 0.01, process_reward: float = 0.05) -> None:
        super().__init__()
        self.failure_penalty = failure_penalty
        self.process_reward = process_reward

    def __call__(self, action: Any, observation: Any) -> float:
        score = self.forward(action, observation)
        return max(0.01, min(0.99, float(score)))

    def forward(self, action: Any, observation: Any) -> float:
        """
        Calculates the reward for the current action.
        """
        done = getattr(observation, "done", False)
        error = getattr(observation, "error", None)

        if error is not None:
            # Penalize invalid actions (wrong tool, missing args)
            return self.failure_penalty

        action_type = getattr(action, "action_type", None)
        
        if done:
            metadata = getattr(observation, "metadata", {})
            
            # Evaluate final outcome
            if action_type == "route_ticket":
                route = getattr(action, "team", "")
                expected = metadata.get("expected_route", "")
                if route and expected and str(route).lower().strip() == str(expected).lower().strip():
                    return 0.99
                return 0.01
                
            elif action_type == "close_ticket":
                res = getattr(action, "resolution", "")
                expected_res = metadata.get("expected_resolution", "")
                if res and expected_res and expected_res.lower() in res.lower():
                    return 0.99
                return 0.01
            
            # Reached max steps without conclusion or random failure
            return self.failure_penalty
            
        # Evaluate intermediate process steps
        if action_type == "query_database":
            # Just querying the valid db gives a tiny reward in intermediate step
            return self.process_reward
            
        if action_type == "ask_clarification":
            return self.process_reward
            
        return 0.01

    def reset(self) -> None:
        pass
