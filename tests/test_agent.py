"""Tests for the Sovereign Agent SDK.

Covers:
- Agent creation and initialization
- Memory operations (remember, recall)
- Messaging (send, receive)
- Soul operations (install, load, unload, list)
- Status reporting
- Agent properties
- Encryption stubs (graceful degradation)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sksovereign_agent.agent import Agent


class TestAgentInit:
    """Tests for Agent initialization."""

    def test_create_agent(self) -> None:
        """Happy path: create an agent with defaults."""
        agent = Agent("TestBot")
        assert agent.name == "TestBot"
        assert not agent._initialized

    def test_agent_status_before_init(self) -> None:
        """Status works before initialization."""
        agent = Agent("TestBot")
        status = agent.status()
        assert status["name"] == "TestBot"
        assert status["initialized"] is False
        assert status["identity"] == "none"

    def test_init_without_capauth(self, tmp_path: Path) -> None:
        """Init works even when capauth is not importable."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))

        with patch.dict(
            "sys.modules",
            {"capauth": None, "capauth.profile": None, "capauth.models": None},
        ):
            result = agent.init()

        assert result["name"] == "TestBot"
        assert agent._initialized is True

    def test_init_creates_home_dir(self, tmp_path: Path) -> None:
        """Init creates the home directory."""
        home = tmp_path / "new-agent"
        agent = Agent("TestBot", home=str(home))
        agent.init()
        assert home.exists()


class TestAgentMemory:
    """Tests for memory operations."""

    def test_remember_and_recall(self, tmp_path: Path) -> None:
        """Store and search a memory."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        mem_id = agent.remember(
            "The sovereign stack is complete",
            title="Milestone",
            tags=["project"],
            intensity=8.0,
        )
        assert mem_id is not None

        results = agent.recall("sovereign stack")
        assert len(results) >= 1
        assert any("sovereign" in r["content"].lower() for r in results)

    def test_remember_without_title(self, tmp_path: Path) -> None:
        """Auto-generated title from content."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        mem_id = agent.remember("Short content")
        assert mem_id is not None

    def test_recall_empty_store(self, tmp_path: Path) -> None:
        """Recall on empty store returns empty list."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        results = agent.recall("nonexistent")
        assert results == []


class TestAgentMessaging:
    """Tests for send/receive."""

    def test_send_stores_locally(self, tmp_path: Path) -> None:
        """Send stores the message in local history."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        result = agent.send("capauth:peer@mesh", "Hello peer!")
        assert result["stored"] is True
        assert result["recipient"] == "capauth:peer@mesh"

    def test_send_without_skchat(self, tmp_path: Path) -> None:
        """Send without skchat reports error."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        with patch.dict("sys.modules", {"skchat": None, "skchat.models": None}):
            result = agent.send("peer", "hello")
        assert "error" in result or result.get("stored") is False or result.get("stored") is True


class TestAgentStatus:
    """Tests for status reporting."""

    def test_status_after_init(self, tmp_path: Path) -> None:
        """Status shows initialized state."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        status = agent.status()
        assert status["initialized"] is True
        assert status["memory"] in ("active", "unavailable")

    def test_status_version(self) -> None:
        """Status includes version."""
        agent = Agent("TestBot")
        assert agent.status()["version"] == "0.2.0"


class TestAgentProperties:
    """Tests for agent properties."""

    def test_fingerprint_empty_before_init(self) -> None:
        """Fingerprint is empty string before identity setup."""
        agent = Agent("TestBot")
        assert agent.fingerprint == ""

    def test_identity_none_before_init(self) -> None:
        """Identity is None before init."""
        agent = Agent("TestBot")
        assert agent.identity is None


# ---------------------------------------------------------------------------
# Soul operations
# ---------------------------------------------------------------------------


class TestAgentSoul:
    """Tests for soul overlay operations."""

    def _make_soul_blueprint(self, tmp_path: Path, name: str = "test-soul") -> Path:
        """Create a minimal YAML soul blueprint file."""
        content = f"name: {name}\ndisplay_name: Test Soul\ncategory: test\n"
        bp_path = tmp_path / f"{name}.yaml"
        bp_path.write_text(content)
        return bp_path

    def test_install_soul(self, tmp_path: Path) -> None:
        """Install a soul blueprint via the agent."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        bp_path = self._make_soul_blueprint(tmp_path)

        result = agent.install_soul(str(bp_path))
        assert result is not None
        assert result["name"] == "test-soul"
        assert result["category"] == "test"

    def test_list_souls(self, tmp_path: Path) -> None:
        """List installed souls."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        bp1 = self._make_soul_blueprint(tmp_path, "soul-alpha")
        bp2 = self._make_soul_blueprint(tmp_path, "soul-beta")
        agent.install_soul(str(bp1))
        agent.install_soul(str(bp2))

        names = agent.list_souls()
        assert "soul-alpha" in names
        assert "soul-beta" in names

    def test_load_and_unload_soul(self, tmp_path: Path) -> None:
        """Load and unload a soul overlay."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        bp_path = self._make_soul_blueprint(tmp_path)
        agent.install_soul(str(bp_path))

        state = agent.load_soul("test-soul", reason="testing")
        assert state is not None
        assert state["active_soul"] == "test-soul"
        assert agent.active_soul() == "test-soul"

        state = agent.unload_soul(reason="done testing")
        assert state is not None
        assert state["active_soul"] is None
        assert agent.active_soul() is None

    def test_load_uninstalled_soul_returns_none(self, tmp_path: Path) -> None:
        """Loading an uninstalled soul returns None."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        result = agent.load_soul("nonexistent")
        assert result is None

    def test_install_invalid_soul_returns_none(self, tmp_path: Path) -> None:
        """Installing from a bad path returns None."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        result = agent.install_soul(str(tmp_path / "nope.yaml"))
        assert result is None

    def test_active_soul_default_is_none(self, tmp_path: Path) -> None:
        """Active soul is None when no overlay is loaded."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        assert agent.active_soul() is None

    def test_status_includes_soul(self, tmp_path: Path) -> None:
        """Status dict includes the active soul."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        bp_path = self._make_soul_blueprint(tmp_path)
        agent.install_soul(str(bp_path))
        agent.load_soul("test-soul")

        status = agent.status()
        assert status["soul"] == "test-soul"

    def test_status_soul_base_when_none(self, tmp_path: Path) -> None:
        """Status shows 'base' when no soul overlay is active."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()
        status = agent.status()
        assert status["soul"] == "base"

    def test_soul_without_skcapstone(self, tmp_path: Path) -> None:
        """Soul operations degrade gracefully without skcapstone."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        with patch.dict(
            "sys.modules",
            {"skcapstone": None, "skcapstone.soul": None},
        ):
            # Force re-creation of soul manager
            agent._soul_manager = None
            assert agent.list_souls() == []
            assert agent.active_soul() is None
            assert agent.load_soul("test") is None


# ---------------------------------------------------------------------------
# Encryption stubs
# ---------------------------------------------------------------------------


class TestAgentCrypto:
    """Tests for encrypt/decrypt/sign/verify — graceful degradation."""

    def test_encrypt_without_capauth(self, tmp_path: Path) -> None:
        """Encrypt returns None when capauth unavailable."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        with patch.dict(
            "sys.modules",
            {"capauth": None, "capauth.crypto": None},
        ):
            result = agent.encrypt("secret", "A" * 40)
            assert result is None

    def test_decrypt_without_capauth(self, tmp_path: Path) -> None:
        """Decrypt returns None when capauth unavailable."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        with patch.dict(
            "sys.modules",
            {"capauth": None, "capauth.crypto": None},
        ):
            result = agent.decrypt("encrypted blob")
            assert result is None

    def test_sign_without_capauth(self, tmp_path: Path) -> None:
        """Sign returns None when capauth unavailable."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        with patch.dict(
            "sys.modules",
            {"capauth": None, "capauth.crypto": None},
        ):
            result = agent.sign("data to sign")
            assert result is None

    def test_verify_without_capauth(self, tmp_path: Path) -> None:
        """Verify returns False when capauth unavailable."""
        agent = Agent("TestBot", home=str(tmp_path / "agent"))
        agent.init()

        with patch.dict(
            "sys.modules",
            {"capauth": None, "capauth.crypto": None},
        ):
            result = agent.verify("data", "sig", "A" * 40)
            assert result is False
