"""Optional intent-aware re-rank of session_search discovery results.

FTS5/BM25 ranks sessions by keyword overlap with the query. On one real install's 37 audited
recall questions, the source session was in the top 5 for 33 but ranked first for only 24. An LLM
that sees the caller's natural-language intent plus the top hits put the source first in all 33
(Opus; 0.649 -> 0.892 hit@1, 9 wins / 0 losses). The same re-rank on the keyword query alone barely
helped (8 wins / 6 losses), so the tool takes an optional ``intent`` and re-ranks only when it is
given AND ``session_search.rerank.enabled`` is on. Off by default: it adds one auxiliary call.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from tools import session_search_tool as sst


@pytest.fixture
def db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    for sid, title, text in (("s_a", "Router firmware", "flashed the router firmware to 4.2"),
                             ("s_b", "Router password", "reset the router admin password"),
                             ("s_c", "Router placement", "moved the router to the hallway")):
        db.create_session(sid, source="cli")
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, sid))
        db.append_message(sid, role="user", content=f"router question: {text}")
        db.append_message(sid, role="assistant", content=f"Done: {text}.")
    db._conn.commit()
    return db


def _ids(out):
    return [r["session_id"] for r in json.loads(out)["results"]]


def _reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_intent_rerank_puts_the_chosen_session_first(db):
    base = _ids(sst.session_search(query="router", limit=3, db=db))
    target = base[-1]
    order = ",".join(str(i) for i in [base.index(target)] + [i for i in range(len(base)) if base[i] != target])
    with patch.object(sst, "_rerank_enabled", return_value=True), \
            patch("agent.auxiliary_client.call_llm", return_value=_reply(order)) as llm:
        out = sst.session_search(query="router", limit=3, db=db, intent="what did I do to the router's admin login?")
    assert _ids(out)[0] == target and sorted(_ids(out)) == sorted(base)
    assert "admin login" in llm.call_args.kwargs["messages"][-1]["content"]  # the intent reaches the re-ranker


@pytest.mark.parametrize("enabled, intent", [(False, "why the admin login?"), (True, "")])
def test_no_rerank_unless_enabled_and_intent_given(db, enabled, intent):
    base = sst.session_search(query="router", limit=3, db=db)
    with patch.object(sst, "_rerank_enabled", return_value=enabled), \
            patch("agent.auxiliary_client.call_llm") as llm:
        out = sst.session_search(query="router", limit=3, db=db, intent=intent)
    llm.assert_not_called()
    assert json.loads(out)["results"] == json.loads(base)["results"]


@pytest.mark.parametrize("llm", [_reply(""), _reply("no idea"), RuntimeError("provider down")])
def test_rerank_failure_keeps_the_fts_order(db, llm):
    base = _ids(sst.session_search(query="router", limit=3, db=db))
    kw = {"side_effect": llm} if isinstance(llm, Exception) else {"return_value": llm}
    with patch.object(sst, "_rerank_enabled", return_value=True), patch("agent.auxiliary_client.call_llm", **kw):
        out = sst.session_search(query="router", limit=3, db=db, intent="the admin login")
    assert _ids(out) == base
