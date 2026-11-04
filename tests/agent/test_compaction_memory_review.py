"""Compaction must not drop turns no memory review has seen.

The background memory review fires every ``memory.nudge_interval`` user turns. Compaction replaces
older turns with a summary, so turns compacted before the next nudge are never reviewed: the next
fork replays the summary, not the turns. On one real install, 77 of 89 compacted sessions compacted
fewer user turns than the nudge interval. ``_pre_compress_memory_context`` (the hook every
compaction passes before it summarises) now spawns the same one-shot review of the unreviewed tail
that session end uses.
"""

import datetime as _dt
import threading

from agent.conversation_compression_checkpoint import _pre_compress_memory_context
from run_agent import AIAgent


def _agent(turns_since_memory):
    agent = object.__new__(AIAgent)
    agent.session_id = "s1"
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 10
    agent._turns_since_memory = turns_since_memory
    agent._memory_manager = None
    agent.skip_background_review = False
    agent._delegate_depth = 0
    agent.session_start = _dt.datetime(2026, 1, 1)
    agent._background_review_lock = threading.Lock()
    agent._background_review_run = None
    agent.spawned = []
    agent._spawn_background_review = lambda **kw: agent.spawned.append(kw)
    return agent


TALK = [{"role": "user", "content": "I run Manjaro on the unstable branch"}, {"role": "assistant", "content": "Noted."}]


def test_compaction_reviews_the_unreviewed_turns_first():
    agent = _agent(4)
    _pre_compress_memory_context(agent, TALK, checkpoint_required=False)
    assert len(agent.spawned) == 1
    assert agent.spawned[0]["messages_snapshot"] == TALK  # the pre-compaction turns, not the summary
    assert agent.spawned[0]["review_memory"] is True and agent.spawned[0]["review_skills"] is False
    assert agent._turns_since_memory == 0


def test_no_review_when_every_turn_was_already_reviewed():
    agent = _agent(0)
    _pre_compress_memory_context(agent, TALK, checkpoint_required=False)
    assert agent.spawned == []
