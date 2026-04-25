"""
Human Proxy Agent for FRIDAY.
Handles user approval for sensitive operations.
"""

from typing import Dict, Any, Optional

class HumanProxy:
    """
    Simulates a human-in-the-loop for AutoGen teams.
    """

    def __init__(self, sandbox_dir: str):
        self.sandbox_dir = sandbox_dir

    def check_approval(self, message: str) -> bool:
        """
        Asks the user for approval if a task involves writing outside the sandbox.
        """
        # In a real CLI environment, this would use 'ask_user' or input()
        # For our agent simulation, we look for path indicators.
        if "sandbox" not in message.lower() and ("write" in message.lower() or "save" in message.lower()):
            print(f"\n[SYSTEM] ATTENTION: Agent is requesting to write to a path outside the sandbox.")
            print(f"[SYSTEM] Context: {message[:100]}...")
            # For automation safety in this environment, we default to False unless sandbox is mentioned.
            return False
        return True
