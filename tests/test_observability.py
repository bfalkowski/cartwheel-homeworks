"""Homework 2, Part D: authentication tests for the session endpoints.

Offline only: no Langfuse, no Docker, no model provider key. Both tests call
the FastAPI route functions directly (as plain Python functions), the same
way tests/test_hw_holes.py::test_hw2_create_session_binds_verified_identity
does, so no server process or HTTP client is needed.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server import app as server_app


@pytest.fixture(autouse=True)
def _clear_sessions(world: dict) -> None:
    """Each test starts with no server-side sessions from a prior test."""
    server_app._SESSIONS.clear()


def test_create_session_rejects_role_that_does_not_match_the_database() -> None:
    """User 1 is a shopper in the seeded world; claiming 'merchant' for them
    must be rejected with 403, and no session may be created for the claim."""
    with pytest.raises(HTTPException) as exc_info:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="merchant"))
    assert exc_info.value.status_code == 403
    assert not server_app._SESSIONS


def test_token_for_one_session_does_not_authorize_another() -> None:
    """A token is bound to the session_id in its signed payload; presenting
    it against a different (also real) session_id must be refused."""
    first = server_app.create_session(server_app.SessionCreate(user_id=1, role="shopper"))
    second = server_app.create_session(
        server_app.SessionCreate(user_id=9002, role="merchant")
    )

    # The token is valid for the session it was issued for...
    ctx = server_app._authorize(first["session_id"], f"Bearer {first['token']}")
    assert ctx.user_id == 1

    # ...but not for a different, equally real session.
    with pytest.raises(HTTPException) as exc_info:
        server_app._authorize(second["session_id"], f"Bearer {first['token']}")
    assert exc_info.value.status_code == 403
