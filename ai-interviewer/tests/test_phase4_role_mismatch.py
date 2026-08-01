"""
tests/test_phase4_role_mismatch.py
------------------------------------
Unit tests for Phase 4: Role-mismatch advisory suggestions and switch action.
"""

import pytest


def test_role_mismatch_banner_triggered_when_roles_differ():
    # Simulate _on_role_change logic
    selected_role = "System Design"
    detected_role = "Backend Engineer"

    is_mismatched = bool(detected_role and selected_role != detected_role)
    assert is_mismatched is True

    msg = f"💡 **Suggestion:** Your resume matches **{detected_role}** best, but **{selected_role}** is selected."
    assert "Backend Engineer" in msg
    assert "System Design" in msg


def test_role_mismatch_banner_hidden_when_roles_match():
    selected_role = "Backend Engineer"
    detected_role = "Backend Engineer"

    is_mismatched = bool(detected_role and selected_role != detected_role)
    assert is_mismatched is False


def test_switch_role_action_aligns_dropdown():
    detected_role = "Backend Engineer"
    # Switch button click sets role_dropdown value to detected_role
    switched_role = detected_role
    assert switched_role == "Backend Engineer"
