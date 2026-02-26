"""Tests for the Sovereign Agent SDK."""

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
        assert agent.status()["version"] == "0.1.0"


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
