"""Quick-start functions -- the simplest possible sovereign agent API.

For developers who want to get started in 3 lines:

    from sovereign_agent import create_identity, store_memory, send_message
    create_identity("MyBot", "bot@example.com", "passphrase")
    store_memory("Learned something important", tags=["project"])
    send_message("peer@mesh", "Hello from sovereign territory!")
"""

from __future__ import annotations

from typing import Optional


def create_identity(
    name: str,
    email: str = "",
    passphrase: str = "",
    entity_type: str = "ai",
) -> dict:
    """Create a sovereign PGP identity.

    Generates a CapAuth profile with PGP keypair.

    Args:
        name: Agent display name.
        email: Email for the PGP UID.
        passphrase: Passphrase for key protection.
        entity_type: human, ai, or organization.

    Returns:
        dict: Identity info with 'name', 'fingerprint', 'email'.
    """
    try:
        from capauth.profile import init_profile
        from capauth.models import EntityType

        profile = init_profile(
            name=name,
            email=email or f"{name.lower()}@sovereign.local",
            passphrase=passphrase or name,
            entity_type=EntityType(entity_type),
        )
        return {
            "name": profile.entity.name,
            "fingerprint": profile.key_info.fingerprint,
            "email": profile.entity.email,
        }
    except ImportError:
        raise ImportError("capauth is required: pip install capauth")


def load_identity(home: str = "~/.capauth") -> dict:
    """Load an existing CapAuth identity.

    Args:
        home: CapAuth home directory.

    Returns:
        dict: Identity info.
    """
    try:
        from capauth.profile import load_profile
        from pathlib import Path

        profile = load_profile(Path(home).expanduser())
        return {
            "name": profile.entity.name,
            "fingerprint": profile.key_info.fingerprint,
            "email": profile.entity.email,
        }
    except ImportError:
        raise ImportError("capauth is required: pip install capauth")


def store_memory(
    content: str,
    title: str = "",
    tags: Optional[list[str]] = None,
    home: str = "~/.skmemory",
) -> str:
    """Store a memory in SKMemory.

    Args:
        content: Memory content.
        title: Short title.
        tags: Searchable tags.
        home: SKMemory home directory.

    Returns:
        str: Memory ID.
    """
    try:
        from pathlib import Path
        from skmemory import MemoryStore, SQLiteBackend

        base = Path(home).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        backend = SQLiteBackend(base_path=str(base))
        store = MemoryStore(primary=backend)

        memory = store.snapshot(
            title=title or content[:60],
            content=content,
            tags=tags or [],
            source="sdk",
        )
        return memory.id
    except ImportError:
        raise ImportError("skmemory is required: pip install skmemory")


def recall_memory(
    query: str,
    limit: int = 5,
    home: str = "~/.skmemory",
) -> list[dict]:
    """Search memories by text.

    Args:
        query: Search query.
        limit: Max results.
        home: SKMemory home directory.

    Returns:
        list[dict]: Matching memories.
    """
    try:
        from pathlib import Path
        from skmemory import MemoryStore, SQLiteBackend

        base = Path(home).expanduser()
        if not base.exists():
            return []

        backend = SQLiteBackend(base_path=str(base))
        store = MemoryStore(primary=backend)
        results = store.search(query, limit=limit)

        return [
            {
                "id": m.id,
                "title": m.title,
                "content": m.content[:200],
                "tags": m.tags,
            }
            for m in results
        ]
    except ImportError:
        raise ImportError("skmemory is required: pip install skmemory")


def send_message(
    recipient: str,
    content: str,
    sender: str = "local",
    thread_id: Optional[str] = None,
) -> dict:
    """Send a chat message via SKChat.

    Args:
        recipient: CapAuth identity URI or agent name.
        content: Message content.
        sender: Sender identity.
        thread_id: Optional thread ID.

    Returns:
        dict: Send result.
    """
    try:
        from skchat.models import ChatMessage
        from skchat.history import ChatHistory
        from skmemory import MemoryStore

        store = MemoryStore()
        history = ChatHistory(store=store)

        msg = ChatMessage(
            sender=sender,
            recipient=recipient,
            content=content,
            thread_id=thread_id,
        )
        mem_id = history.store_message(msg)

        return {
            "stored": True,
            "memory_id": mem_id,
            "message_id": msg.id,
            "recipient": recipient,
        }
    except ImportError:
        raise ImportError("skchat and skmemory are required")
