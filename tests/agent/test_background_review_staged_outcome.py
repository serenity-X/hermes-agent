"""The review completion log must say when a review STAGED writes for approval.

With ``memory.write_approval`` / ``skills.write_approval`` on, a review fork's writes return
``{"success": true, "staged": true, ...}`` and are deliberately kept out of the user-facing summary,
so ``_classify_review_result`` saw no action and logged ``result=none``. On one real install 61 of 74
staged skill-write records came from reviews whose completion line said ``result=none``, so the log
could not tell "the review did nothing" from "the review queued work for you to approve".
"""

import json

import agent.background_review as bg


def _review(tool, payload, args):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(payload)},
    ]


def test_staged_skill_write_is_logged_as_staged():
    msgs = _review("skill_manage", {"success": True, "staged": True, "pending_id": "abc123", "message": "staged"},
                   {"operations": [{"action": "patch", "name": "github"}]})
    assert bg.summarize_background_review_actions(msgs, [], "on") == []  # user-facing summary unchanged
    assert bg._review_outcome(msgs, [], []) == "staged"


def test_applied_write_keeps_its_result_and_a_quiet_review_stays_none():
    applied = ["Memory updated"]
    assert bg._review_outcome([], [], applied) == bg._classify_review_result(applied)
    assert bg._review_outcome([], [], []) == "none"
