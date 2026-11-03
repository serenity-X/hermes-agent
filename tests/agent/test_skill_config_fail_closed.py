"""A failed read of config.yaml must not re-enable every disabled skill.

``_load_raw_config`` returned ``{}`` when the parse raised, so ``skills.disabled`` read as empty and
every disabled skill came back into the index for that call. A config caught mid-write (an editor's
truncate-then-write, a half-synced file) triggers it. It now serves the last good parse of the same
config path instead. OpenHands shipped the same fix after an incident.
"""

from agent import skill_utils


def _write(path, disabled):
    path.write_text("skills:\n  disabled:\n" + "".join(f"    - {n}\n" for n in disabled), encoding="utf-8")


def test_unparsable_config_keeps_the_last_good_disabled_list(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"; home.mkdir()
    config = home / "config.yaml"
    _write(config, ["airtable", "arxiv"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    skill_utils._raw_config_cache_clear()
    assert skill_utils.get_disabled_skill_names() == {"airtable", "arxiv"}

    config.write_text("skills:\n  disabled: [airtable,\n", encoding="utf-8")  # caught mid-write
    assert skill_utils.get_disabled_skill_names() == {"airtable", "arxiv"}


def test_first_read_failure_still_returns_empty(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"; home.mkdir()
    (home / "config.yaml").write_text("skills:\n  disabled: [airtable,\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    skill_utils._raw_config_cache_clear()
    assert skill_utils.get_disabled_skill_names() == set()
