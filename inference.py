import os
import json
import uuid
from openai import OpenAI
from server.environment import SupportEnvironment
from models import SupportAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required. Please set it to proceed.")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# A simplified schema of SupportAction for the LLM to use
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "take_action",
            "description": "Take an action to handle the support ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["route_ticket", "ask_clarification", "query_database", "close_ticket"],
                        "description": "The type of action to take."
                    },
                    "team": {
                        "type": "string",
                        "description": "The support team to route to (if routing)."
                    },
                    "question": {
                        "type": "string",
                        "description": "The clarifying question to ask the user (if asking clarification)."
                    },
                    "query_customer_id": {
                        "type": "string",
                        "description": "The customer ID to query in the DB."
                    },
                    "resolution": {
                        "type": "string",
                        "description": "How the ticket was resolved."
                    }
                },
                "required": ["action_type"]
            }
        }
    }
]

def run_episode(env, episode_id):
    obs = env.reset(episode_id=episode_id)
    task_difficulty = env.state.current_task_difficulty
    benchmark = "support-triage-v1"
    
    print(f"[START] task={task_difficulty} env={benchmark} model={MODEL_NAME}")
    
    messages = [
        {"role": "system", "content": "You are a customer support agent. Resolve the user's issue or route it to the right team. Available teams usually involve IT, General, Billing, Premium_Billing, Hardware_Returns."},
        {"role": "user", "content": f"New Ticket ID {obs.ticket_id}:\n{obs.ticket_text}\nCustomer ID (if any): {obs.customer_id}"}
    ]
    
    rewards_history = []
    
    # We allow up to 6 steps per ticket
    for step in range(1, 7):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
        except Exception as e:
            err_msg = str(e).replace("'", "").replace("\n", " ")
            print(f"[STEP] step={step} action=none reward=0.00 done=true error='API Error: {err_msg}'")
            rewards_str = ",".join(rewards_history + ["0.00"])
            print(f"[END] success=false steps={step} rewards={rewards_str}")
            return
        
        message = response.choices[0].message
        
        # If model decides to use the tool
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                action_args = json.loads(tool_call.function.arguments)
            except Exception as e:
                err_msg = str(e).replace("'", "").replace("\n", " ")
                print(f"[STEP] step={step} action=none reward=0.00 done=true error='JSON Parse Error: {err_msg}'")
                rewards_str = ",".join(rewards_history + ["0.00"])
                print(f"[END] success=false steps={step} rewards={rewards_str}")
                return
            
            # Map LLM's raw dict to our typed Pydantic structure
            action = SupportAction(
                action_type=action_args.get("action_type"),
                team=action_args.get("team"),
                question=action_args.get("question"),
                query_customer_id=action_args.get("query_customer_id"),
                resolution=action_args.get("resolution")
            )
            action_str = f"action_type='{action.action_type}'"
            
            messages.append(message)
            
            # Step the environment
            next_obs = env.step(action)
            current_reward = getattr(next_obs, "reward", 0.0)
            rewards_history.append(f"{current_reward:.2f}")
            
            err = getattr(next_obs, 'error', None)
            done = getattr(next_obs, 'done', False)
            err_str = f"'{err}'" if err else "null"
            done_str = "true" if done else "false"
            
            print(f"[STEP] step={step} action={action_str} reward={current_reward:.2f} done={done_str} error={err_str}")
            
            if done:
                success = "true" if current_reward >= 0.9 else "false"
                rewards_str = ",".join(rewards_history)
                print(f"[END] success={success} steps={step} rewards={rewards_str}")
                return
                
            # Feed result back for next iteration
            tool_text = "Action succeeded."
            if err:
                tool_text = f"Error: {err}"
            elif next_obs.db_result:
                tool_text = f"DB Result: {next_obs.db_result}"
            elif next_obs.clarification_reply:
                tool_text = f"User Replied: {next_obs.clarification_reply}"
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_text
            })
            
        else:
            # Model didn't use a tool, force a stop or tell it to act
            print(f"[STEP] step={step} action=none reward=0.00 done=true error='Model failed to select tool'")
            rewards_str = ",".join(rewards_history + ["0.00"])
            print(f"[END] success=false steps={step} rewards={rewards_str}")
            return
            
    # Max steps reached without resolving
    rewards_str = ",".join(rewards_history)
    print(f"[END] success=false steps=6 rewards={rewards_str}")

def main():
    env = SupportEnvironment()
    
    # Run 3 episodes to simulate easy, medium, hard
    for i in range(3):
        # We manually cycle the seed so it selects different tasks deterministically if needed,
        # but SupportEnvironment uses random.choice without tracking index. We'll rely on it.
        ep_id = str(uuid.uuid4())
        run_episode(env, ep_id)

if __name__ == "__main__":
    main()
