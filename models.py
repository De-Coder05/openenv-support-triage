from typing import Any, Dict, Literal, Optional
from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone support
    from openenv.core.env_server.types import Action, Observation, State


class SupportAction(Action):
    """
    Action type to handle multi-turn operations in the Support Ticket env.
    Instead of unions, all fields are collected here and utilized based on action_type.
    """
    action_type: Literal["route_ticket", "ask_clarification", "query_database", "close_ticket"] = Field(
        ..., description="The type of action to execute"
    )
    team: Optional[str] = Field(None, description="Team to route to (for route_ticket)")
    question: Optional[str] = Field(None, description="Clarifying question (for ask_clarification)")
    query_customer_id: Optional[str] = Field(None, description="Customer ID to look up (for query_database)")
    resolution: Optional[str] = Field(None, description="Summary resolution (for close_ticket)")


class SupportObservation(Observation):
    """
    Observation type containing the context from the environment simulation.
    """
    ticket_id: str = Field(..., description="Unique ID of the current ticket")
    ticket_text: str = Field(..., description="The customer's email or support request content.")
    customer_id: Optional[str] = Field(None, description="The customer ID parsed from the ticket (if available).")
    
    # Contexts added through multi-turn actions
    db_result: Optional[str] = Field(None, description="Result fetched from a 'query_database' action.")
    clarification_reply: Optional[str] = Field(None, description="Customer reply after an 'ask_clarification' action.")
    error: Optional[str] = Field(None, description="Error message if the last action was invalid or unsupported.")


class SupportState(State):
    """
    Internal environment state. Tracks ground truths and difficulty progressions.
    """
    expected_route: Optional[str] = Field(None, description="Ground truth answer for grading")
    customer_tier: Optional[str] = Field(None, description="Customer tier for DB simulation (Medium/Expert)")
    clarification_asked: bool = Field(False, description="Has clarification been asked?")
    task_difficulty: str = Field("easy", description="Difficulty of the current task instance")
