"""The inline ``session_search`` executor (the path a live agent turn uses) must forward ``intent``.

The agent loop runs session_search through ``agent/inline_tool_executors.py``, which maps a fixed
argument list onto the tool, not through the registry handler. Without ``intent`` in that list a
model that fills the field is ignored and the opt-in re-rank never sees it.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext


def test_inline_executor_forwards_intent():
    agent = SimpleNamespace(_get_session_db_for_recall=lambda: object(), session_id="s1")
    ctx = InlineToolContext(effective_task_id="t1", tool_call_id="c1")
    with patch("tools.session_search_tool.session_search", return_value=json.dumps({"success": True})) as tool:
        INLINE_TOOL_EXECUTORS["session_search"](agent, {"query": "router", "intent": "the admin login"}, ctx)
    assert tool.call_args.kwargs["intent"] == "the admin login"
