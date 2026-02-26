"""Sovereign Agent SDK — build AI agents with sovereign identity and memory.

One package. Full stack. No corporate middleman.

    pip install sksovereign-agent

This SDK re-exports the key APIs from the sovereign stack:
- CapAuth: PGP identity, challenge-response auth, capability tokens
- SKMemory: persistent memory with emotional context
- SKChat: encrypted P2P messaging with group chat
- SKComm: transport-agnostic message delivery
- Cloud 9: emotional continuity protocol

Quick start:
    from sksovereign_agent import Agent
    agent = Agent("MyAgent")
    agent.init()
    agent.send("peer@mesh", "Hello from the sovereign side!")
"""

__version__ = "0.1.0"
__author__ = "smilinTux Team"
__license__ = "GPL-3.0-or-later"

from .agent import Agent
from .quick import (
    create_identity,
    load_identity,
    send_message,
    store_memory,
    recall_memory,
)

__all__ = [
    "Agent",
    "create_identity",
    "load_identity",
    "send_message",
    "store_memory",
    "recall_memory",
    "__version__",
]
