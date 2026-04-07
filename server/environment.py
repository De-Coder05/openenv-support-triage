import random
from typing import Any, Optional
import uuid

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = False
        def __init__(self, *args, **kwargs): pass
        def reset(self, *args, **kwargs): raise NotImplementedError
        def step(self, *args, **kwargs): raise NotImplementedError
        @property
        def state(self): raise NotImplementedError
        def get_metadata(self): return None
        def _apply_rubric(self, action, observation): return 0.0

from models import SupportAction, SupportObservation, SupportState
from .rubrics import SupportRubric


# Some mock DB logic for our medium/hard tasks
MOCK_DB = {
    "C-7788": "Customer Tier: Enterprise",
    "C-1010": "Customer Tier: Standard"
}

# The hardcoded tasks simulating a Support Ticket queue
TASKS = [
    {
        "difficulty": "easy",
        "ticket_text": "I can't log in to my account. Please reset my password. Thank you.",
        "customer_id": None,
        "expected_route": "IT",
    },
    {
        "difficulty": "easy",
        "ticket_text": "How do I update my profile picture?",
        "customer_id": None,
        "expected_route": "General",
    },
    {
        "difficulty": "medium",
        "ticket_text": "My payment for invoice #1234 failed. Please help. My customer ID is C-7788.",
        "customer_id": "C-7788",
        "expected_route": "Premium_Billing",  # Because tier is Enterprise
    },
    {
        "difficulty": "medium",
        "ticket_text": "Double charge on my credit card. My customer ID is C-1010.",
        "customer_id": "C-1010",
        "expected_route": "Billing", # Standard tier
    },
    {
        "difficulty": "hard",
        "ticket_text": "I want to return the product I bought last week.",
        "customer_id": None,
        "expected_route": "Hardware_Returns",
        "clarification_reply_mock": "I bought the physical Ultra-Widget 3000 device."
    }
]

class SupportEnvironment(Environment):
    """
    Support Triage Environment implementing the OpenEnv Environment interface.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.rubric = SupportRubric()
        
        # Internal tracking variables reset per episode
        self._state = SupportState(step_count=0)
        self._current_task_info = {}
        self._ticket_id = ""

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> SupportObservation:
        self.rubric.reset()
        
        if seed is not None:
            random.seed(seed)
            
        task = random.choice(TASKS)
        self._current_task_info = task
        self._ticket_id = f"TKT-{random.randint(1000, 9999)}"
        
        self._state = SupportState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_difficulty=task["difficulty"],
            expected_route=task["expected_route"],
            customer_tier="Enterprise" if task.get("customer_id") == "C-7788" else "Standard",
            clarification_asked=False
        )
        
        obs = SupportObservation(
            ticket_id=self._ticket_id,
            ticket_text=task["ticket_text"],
            customer_id=task.get("customer_id"),
            done=False,
            metadata={"expected_route": task["expected_route"]}
        )
        # Give initial reward = 0
        obs.reward = 0.0
        self._last_obs = obs
        return obs

    def step(self, action: SupportAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SupportObservation:
        self._state.step_count += 1
        
        obs = SupportObservation(
            ticket_id=self._ticket_id,
            ticket_text=self._current_task_info["ticket_text"],
            customer_id=self._current_task_info.get("customer_id"),
            done=False,
            metadata={"expected_route": self._current_task_info["expected_route"]}
        )

        # Process the action
        if action.action_type == "query_database":
            if not action.query_customer_id:
                obs.error = "Missing customer_id for query_database."
            else:
                db_res = MOCK_DB.get(action.query_customer_id, "Customer not found.")
                obs.db_result = db_res
                
        elif action.action_type == "ask_clarification":
            if not action.question:
                obs.error = "Missing question for ask_clarification."
            else:
                if self._current_task_info["difficulty"] == "hard":
                    obs.clarification_reply = self._current_task_info.get("clarification_reply_mock", "I don't know.")
                    self._state.clarification_asked = True
                else:
                    obs.clarification_reply = "I don't have any more info, please just help."
        
        elif action.action_type == "route_ticket":
            if not action.team:
                obs.error = "Missing team for route_ticket."
            else:
                # Terminal step
                obs.done = True
                obs.metadata["actual_route"] = action.team
                
        elif action.action_type == "close_ticket":
            # Terminal step
            obs.done = True
            
        else:
            obs.error = f"Unknown action: {action.action_type}"
            
        # Hard task constraint checker: if they rout without asking context first!
        if self._current_task_info["difficulty"] == "hard" and action.action_type == "route_ticket":
            if not self._state.clarification_asked:
                obs.error = "Cannot determine route without clarifying the product type."
                obs.done = False # Intervene and don't let them close it wrong, force explicit failure

        # Cap iterations
        if self._state.step_count > 10 and not obs.done:
            obs.done = True
            obs.error = "Exceeded maximum iterations (10)."

        # Apply rubric using the built-in Environment rubric system
        # self.rubric handles positive process rew for ask_clarification/query_db and outcome rew
        obs.reward = self._apply_rubric(action, obs)

        self._last_obs = obs
        return obs

    @property
    def state(self) -> SupportState:
        return self._state
