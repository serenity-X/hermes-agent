"""``/context`` must count the skills index where the prompt builder puts it.

``build_system_prompt_parts`` places the ``<available_skills>`` block in the VOLATILE tier
(``agent/system_prompt.py``: ``volatile_parts = [skills_prompt, *memory parts]``), but the breakdown
looked for it in the stable tier only. On a real prompt the ``skills`` category read 0 tokens and the
index was counted as ``system_prompt``. The older test passed only because its fixture put the block in
the stable tier.
"""

from unittest.mock import MagicMock, patch

from agent.context_breakdown import compute_context_details, compute_session_context_breakdown

SKILLS = "<available_skills>\n  demo:\n    - hello: say hi to the user in a friendly way\n</available_skills>"


def _agent():
    agent = MagicMock()
    agent.model = "openai/gpt-5.4"
    agent.tools = [{"type": "function", "function": {"name": "terminal", "description": "run"}}]
    agent._memory_store = None
    agent.context_compressor = MagicMock(context_length=200_000, last_prompt_tokens=0)
    return agent


def test_skills_index_in_the_volatile_tier_is_counted_as_skills():
    parts = {"stable": "identity and guidance", "context": "", "volatile": SKILLS + "\n\nCurrent time: now"}
    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(_agent(), [])
    cats = {c["id"]: c for c in data["categories"]}
    assert cats["skills"]["tokens"] > 0
    with patch("agent.system_prompt.build_system_prompt_parts", return_value={**parts, "volatile": "Current time: now"}):
        without = {c["id"]: c for c in compute_session_context_breakdown(_agent(), [])["categories"]}
    assert cats["system_prompt"]["tokens"] == without["system_prompt"]["tokens"]


def test_context_all_lists_skills_from_the_volatile_tier():
    parts = {"stable": "identity", "context": "", "volatile": SKILLS}
    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        detail = compute_context_details(_agent())
    assert [s["name"] for s in detail["skills"]] == ["hello"]
