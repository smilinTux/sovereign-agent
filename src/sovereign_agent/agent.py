"""High-level Agent class -- the unified entry point for sovereign AI.

The Agent wraps identity, memory, messaging, and transport into
a single object. Third-party developers interact with this instead
of managing individual packages.

Usage:
    agent = Agent("Jarvis", home="~/.skcapstone")
    agent.init(email="jarvis@skworld.io", passphrase="secret")
    agent.remember("Learned about sovereignty", tags=["important"])
    agent.send("peer@mesh", "Hello!")
    messages = agent.receive()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("sovereign_agent")


class Agent:
    """A sovereign AI agent with identity, memory, and communication.

    This is the high-level SDK entry point. It lazily initializes
    each subsystem on first use, so you only pay for what you need.

    Args:
        name: Agent display name.
        home: Agent home directory (default: ~/.skcapstone).
    """

    def __init__(self, name: str = "sovereign-agent", home: str = "~/.skcapstone") -> None:
        self.name = name
        self.home = Path(home).expanduser()
        self._identity: Optional[dict] = None
        self._memory_store: Any = None
        self._history: Any = None
        self._initialized = False

    def init(
        self,
        email: str = "",
        passphrase: str = "",
        entity_type: str = "ai",
    ) -> dict:
        """Initialize the agent: create identity + memory store.

        Generates a CapAuth PGP identity if one doesn't exist,
        and sets up the SKMemory store.

        Args:
            email: Email for the PGP identity.
            passphrase: Passphrase for the PGP key.
            entity_type: human, ai, or organization.

        Returns:
            dict: Initialization summary.
        """
        result = {"name": self.name, "home": str(self.home)}

        self.home.mkdir(parents=True, exist_ok=True)

        try:
            from capauth.profile import init_profile, load_profile
            from capauth.models import EntityType, Algorithm

            capauth_dir = self.home / "capauth"
            try:
                profile = load_profile(capauth_dir)
                self._identity = {
                    "name": profile.entity.name,
                    "fingerprint": profile.key_info.fingerprint,
                    "email": profile.entity.email,
                }
                result["identity"] = "loaded"
            except Exception:
                if passphrase:
                    profile = init_profile(
                        name=self.name,
                        email=email or f"{self.name.lower()}@sovereign.local",
                        passphrase=passphrase,
                        entity_type=EntityType(entity_type),
                        base_dir=capauth_dir,
                    )
                    self._identity = {
                        "name": profile.entity.name,
                        "fingerprint": profile.key_info.fingerprint,
                        "email": profile.entity.email,
                    }
                    result["identity"] = "created"
                else:
                    result["identity"] = "skipped (no passphrase)"
        except ImportError:
            result["identity"] = "capauth not installed"

        try:
            self._init_memory()
            result["memory"] = "ready"
        except ImportError:
            result["memory"] = "skmemory not installed"

        self._initialized = True
        logger.info("Agent %s initialized: %s", self.name, result)
        return result

    @property
    def identity(self) -> Optional[dict]:
        """The agent's identity info (name, fingerprint, email).

        Returns:
            Optional[dict]: Identity dict, or None if not initialized.
        """
        return self._identity

    @property
    def fingerprint(self) -> str:
        """The agent's PGP fingerprint.

        Returns:
            str: 40-char hex fingerprint, or empty string.
        """
        if self._identity:
            return self._identity.get("fingerprint", "")
        return ""

    def remember(
        self,
        content: str,
        title: str = "",
        tags: Optional[list[str]] = None,
        intensity: float = 0.0,
    ) -> Optional[str]:
        """Store a memory in SKMemory.

        Args:
            content: The memory content.
            title: Short title (auto-generated if empty).
            tags: Searchable tags.
            intensity: Emotional intensity (0-10).

        Returns:
            Optional[str]: Memory ID, or None if memory not available.
        """
        store = self._get_memory()
        if store is None:
            return None

        from skmemory.models import EmotionalSnapshot

        title = title or content[:60]
        emotional = EmotionalSnapshot(intensity=intensity) if intensity > 0 else None

        memory = store.snapshot(
            title=title,
            content=content,
            tags=tags or [],
            emotional=emotional,
            source="sdk",
        )
        return memory.id

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Search memories by text.

        Args:
            query: Search query.
            limit: Max results.

        Returns:
            list[dict]: Matching memories as dicts.
        """
        store = self._get_memory()
        if store is None:
            return []

        results = store.search(query, limit=limit)
        return [
            {
                "id": m.id,
                "title": m.title,
                "content": m.content[:200],
                "tags": m.tags,
                "intensity": m.emotional.intensity,
            }
            for m in results
        ]

    def send(
        self,
        recipient: str,
        content: str,
        thread_id: Optional[str] = None,
    ) -> dict:
        """Send a chat message to another agent.

        Uses SKChat + SKComm if available, falls back to
        local storage only.

        Args:
            recipient: CapAuth identity URI or agent name.
            content: Message content.
            thread_id: Optional thread identifier.

        Returns:
            dict: Send result with 'stored' and optionally 'delivered'.
        """
        result: dict[str, Any] = {"recipient": recipient, "stored": False}

        try:
            from skchat.models import ChatMessage

            sender = self._identity.get("fingerprint", self.name) if self._identity else self.name
            msg = ChatMessage(
                sender=f"capauth:{sender}",
                recipient=recipient,
                content=content,
                thread_id=thread_id,
            )

            history = self._get_history()
            if history:
                history.store_message(msg)
                result["stored"] = True

            try:
                from skchat.transport import ChatTransport
                from skcomm import SKComm

                comm = SKComm.from_config()
                transport = ChatTransport(skcomm=comm, history=history, identity=f"capauth:{sender}")
                delivery = transport.send_message(msg)
                result["delivered"] = delivery.get("delivered", False)
            except Exception:
                result["delivered"] = False

        except ImportError:
            result["error"] = "skchat not installed"

        return result

    def receive(self) -> list[dict]:
        """Poll for incoming messages.

        Returns:
            list[dict]: Received messages as dicts.
        """
        try:
            from skchat.transport import ChatTransport
            from skcomm import SKComm

            history = self._get_history()
            if not history:
                return []

            comm = SKComm.from_config()
            sender = self._identity.get("fingerprint", self.name) if self._identity else self.name
            transport = ChatTransport(skcomm=comm, history=history, identity=f"capauth:{sender}")
            messages = transport.poll_inbox()

            return [
                {
                    "sender": m.sender,
                    "content": m.content,
                    "thread_id": m.thread_id,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in messages
            ]
        except ImportError:
            return []

    def status(self) -> dict:
        """Get the agent's current status.

        Returns:
            dict: Status summary.
        """
        s: dict[str, Any] = {
            "name": self.name,
            "home": str(self.home),
            "initialized": self._initialized,
            "version": "0.1.0",
        }

        if self._identity:
            s["fingerprint"] = self._identity.get("fingerprint", "")[:16] + "..."
            s["identity"] = "active"
        else:
            s["identity"] = "none"

        store = self._get_memory()
        if store:
            try:
                mems = store.list_memories(limit=1)
                s["memory"] = "active"
            except Exception:
                s["memory"] = "error"
        else:
            s["memory"] = "unavailable"

        return s

    def _init_memory(self) -> None:
        """Initialize the SKMemory store."""
        from skmemory import MemoryStore, SQLiteBackend

        mem_path = self.home / "memory"
        mem_path.mkdir(parents=True, exist_ok=True)
        backend = SQLiteBackend(base_path=str(mem_path))
        self._memory_store = MemoryStore(primary=backend)

    def _get_memory(self) -> Any:
        """Get or create the memory store.

        Returns:
            MemoryStore or None.
        """
        if self._memory_store is None:
            try:
                self._init_memory()
            except ImportError:
                return None
        return self._memory_store

    def _get_history(self) -> Any:
        """Get or create the ChatHistory.

        Returns:
            ChatHistory or None.
        """
        if self._history is not None:
            return self._history

        store = self._get_memory()
        if store is None:
            return None

        try:
            from skchat.history import ChatHistory

            self._history = ChatHistory(store=store)
            return self._history
        except ImportError:
            return None
