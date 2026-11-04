"""A session that ends before the turn-based nudge still gets one memory review of its tail.

The background memory review fires only every ``memory.nudge_interval`` user turns (default 10), and
no teardown path reviewed the remainder. On one real install 131 of 145 top-level sessions in 30 days
ended with fewer than 10 user turns, so most sessions were never reviewed; 22 of 48 hand-labelled
should-save facts sat in such sessions. ``shutdown_memory_provider`` now spawns one memory-only review
when unreviewed user turns remain (prior art: #62046, #31609).
"""

import datetime as _dt
import threading

import pytest

from run_agent import AIAgent


def _agent(turns_since_memory: int, messages):
    agent = object.__new__(AIAgent)
    agent.session_id = "s1"
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 10
    agent._turns_since_memory = turns_since_memory
    agent._session_messages = messages
    agent._memory_manager = None
    agent._memory_provider_shutdown = False
    agent.skip_background_review = False
    agent._delegate_depth = 0
    agent.session_start = _dt.datetime(2026, 1, 1)
    agent._background_review_lock = threading.Lock()
    agent._background_review_run = None
    agent.spawned = []
    agent._spawn_background_review = lambda **kw: agent.spawned.append(kw)
    return agent


TALK = [{"role": "user", "content": "I run Manjaro on the unstable branch"}, {"role": "assistant", "content": "Noted."}]


def test_unreviewed_tail_gets_one_memory_review_at_shutdown(monkeypatch):
    monkeypatch.setattr("run_agent._notify_context_engine_session_end", lambda *a, **k: None)
    agent = _agent(3, TALK)
    agent.shutdown_memory_provider(TALK)
    agent.shutdown_memory_provider(TALK)  # idempotent: a second teardown path must not review again
    assert len(agent.spawned) == 1
    call = agent.spawned[0]
    assert call["review_memory"] is True and call["review_skills"] is False
    assert call["messages_snapshot"] == TALK
    assert agent._turns_since_memory == 0


@pytest.mark.parametrize("turns, messages, skip", [
    (0, TALK, False),                                   # the periodic review already covered every turn
    (3, [{"role": "assistant", "content": "hi"}], False),  # no user turn: nothing to learn from
    (3, TALK, True),                                    # cron / skip_background_review sessions stay unreviewed
])
def test_no_session_end_review_when_nothing_is_pending(monkeypatch, turns, messages, skip):
    monkeypatch.setattr("run_agent._notify_context_engine_session_end", lambda *a, **k: None)
    agent = _agent(turns, messages)
    agent.skip_background_review = skip
    agent.shutdown_memory_provider(messages)
    assert agent.spawned == []
