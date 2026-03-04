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
        self._soul_manager: Any = None
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

    # ------------------------------------------------------------------
    # Soul operations
    # ------------------------------------------------------------------

    def load_soul(self, name: str, reason: str = "") -> Optional[dict]:
        """Load a soul overlay by name.

        Activates a previously installed soul blueprint, changing the
        agent's personality without changing its identity.

        Args:
            name: Soul slug name (e.g. "the-developer").
            reason: Why the soul is being loaded.

        Returns:
            Optional[dict]: Soul state dict, or None if skcapstone unavailable.
        """
        mgr = self._get_soul_manager()
        if mgr is None:
            return None
        try:
            state = mgr.load(name, reason=reason)
            return state.model_dump()
        except ValueError as exc:
            logger.warning("Failed to load soul '%s': %s", name, exc)
            return None

    def unload_soul(self, reason: str = "") -> Optional[dict]:
        """Unload the active soul overlay, returning to base.

        Args:
            reason: Why the soul is being unloaded.

        Returns:
            Optional[dict]: Updated soul state, or None.
        """
        mgr = self._get_soul_manager()
        if mgr is None:
            return None
        state = mgr.unload(reason=reason)
        return state.model_dump()

    def install_soul(self, path: str) -> Optional[dict]:
        """Install a soul blueprint from a file.

        Args:
            path: Path to .md, .yaml, or .yml blueprint file.

        Returns:
            Optional[dict]: Installed soul blueprint info, or None.
        """
        mgr = self._get_soul_manager()
        if mgr is None:
            return None
        try:
            bp = mgr.install(Path(path))
            return {
                "name": bp.name,
                "display_name": bp.display_name,
                "category": bp.category,
                "traits": len(bp.core_traits),
            }
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Failed to install soul from %s: %s", path, exc)
            return None

    def list_souls(self) -> list[str]:
        """List installed soul names.

        Returns:
            list[str]: Sorted list of installed soul slug names.
        """
        mgr = self._get_soul_manager()
        if mgr is None:
            return []
        return mgr.list_installed()

    def active_soul(self) -> Optional[str]:
        """Get the currently active soul overlay name.

        Returns:
            Optional[str]: Soul name, or None if at base.
        """
        mgr = self._get_soul_manager()
        if mgr is None:
            return None
        return mgr.get_active_soul_name()

    # ------------------------------------------------------------------
    # Encryption operations
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str, recipient_fingerprint: str) -> Optional[str]:
        """Encrypt a message for a recipient using their PGP public key.

        Args:
            plaintext: The message to encrypt.
            recipient_fingerprint: PGP fingerprint of the recipient.

        Returns:
            Optional[str]: PGP-encrypted message armor, or None on failure.
        """
        try:
            from capauth.crypto import get_backend

            backend = get_backend()
            capauth_dir = self.home / "capauth"
            pub_key = backend.load_public_key(capauth_dir, recipient_fingerprint)
            if pub_key is None:
                logger.warning("Public key not found for %s", recipient_fingerprint)
                return None
            return backend.encrypt(plaintext.encode("utf-8"), pub_key)
        except ImportError:
            logger.warning("capauth not installed — cannot encrypt")
            return None
        except Exception as exc:
            logger.warning("Encryption failed: %s", exc)
            return None

    def decrypt(self, ciphertext: str, passphrase: str = "") -> Optional[str]:
        """Decrypt a PGP-encrypted message using the agent's private key.

        Args:
            ciphertext: PGP-encrypted armor string.
            passphrase: Passphrase to unlock the private key.

        Returns:
            Optional[str]: Decrypted plaintext, or None on failure.
        """
        try:
            from capauth.crypto import get_backend

            backend = get_backend()
            capauth_dir = self.home / "capauth"
            priv_key = backend.load_private_key(capauth_dir, passphrase=passphrase)
            if priv_key is None:
                logger.warning("Private key not found or passphrase incorrect")
                return None
            raw = backend.decrypt(ciphertext, priv_key)
            return raw.decode("utf-8") if isinstance(raw, bytes) else raw
        except ImportError:
            logger.warning("capauth not installed — cannot decrypt")
            return None
        except Exception as exc:
            logger.warning("Decryption failed: %s", exc)
            return None

    def sign(self, data: str) -> Optional[str]:
        """Sign data with the agent's PGP private key.

        Args:
            data: The data to sign.

        Returns:
            Optional[str]: PGP signature armor, or None.
        """
        try:
            from capauth.crypto import get_backend

            backend = get_backend()
            capauth_dir = self.home / "capauth"
            priv_key = backend.load_private_key(capauth_dir)
            if priv_key is None:
                return None
            return backend.sign(data.encode("utf-8"), priv_key)
        except (ImportError, Exception) as exc:
            logger.warning("Signing failed: %s", exc)
            return None

    def verify(self, data: str, signature: str, signer_fingerprint: str) -> bool:
        """Verify a PGP signature.

        Args:
            data: The original data that was signed.
            signature: PGP signature armor.
            signer_fingerprint: Expected signer's PGP fingerprint.

        Returns:
            bool: True if signature is valid.
        """
        try:
            from capauth.crypto import get_backend

            backend = get_backend()
            capauth_dir = self.home / "capauth"
            pub_key = backend.load_public_key(capauth_dir, signer_fingerprint)
            if pub_key is None:
                return False
            return backend.verify(data.encode("utf-8"), signature, pub_key)
        except (ImportError, Exception) as exc:
            logger.warning("Verification failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Get the agent's current status.

        Returns:
            dict: Status summary.
        """
        s: dict[str, Any] = {
            "name": self.name,
            "home": str(self.home),
            "initialized": self._initialized,
            "version": "0.2.0",
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

        soul = self.active_soul()
        s["soul"] = soul or "base"

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

    def _get_soul_manager(self) -> Any:
        """Get or create the SoulManager.

        Returns:
            SoulManager or None.
        """
        if self._soul_manager is not None:
            return self._soul_manager

        try:
            from skcapstone.soul import SoulManager

            self._soul_manager = SoulManager(self.home)
            return self._soul_manager
        except ImportError:
            return None
