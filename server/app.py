"""
FastAPI application for the Support Triage Environment.

Creates the HTTP server that exposes SupportEnvironment using the standard
OpenEnv `create_app` utility.
"""

import logging

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    # Fallback missing OpenEnv
    def create_app(*args, **kwargs):
        from fastapi import FastAPI
        return FastAPI()

from models import SupportAction, SupportObservation
from .environment import SupportEnvironment


_logger = logging.getLogger(__name__)

def create_support_environment() -> SupportEnvironment:
    """Factory function for the OpenEnv HTTP Server to instantiate."""
    return SupportEnvironment()


# Create the standard OpenEnv application
app = create_app(
    create_support_environment,
    SupportAction,
    SupportObservation,
    env_name="support-triage-v1",
    max_concurrent_envs=8,
)

def main():
    """Entry point for local execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
