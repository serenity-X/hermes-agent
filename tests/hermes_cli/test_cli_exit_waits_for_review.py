"""The CLI's session-end memory review must finish before the process exits.

``shutdown_memory_provider`` spawns the session-end review on a daemon thread. Both CLI exit paths
(the REPL's ``_run_cleanup`` and the one-shot finalize) call ``_shutdown_agent_memory_provider`` and
then exit, which kills the thread: in a real-model run the review never made its first request
(``os._exit``) or never finished it (normal interpreter exit). The CLI now waits for the review's
request to end, bounded; the exit watchdog stays the backstop.
"""

import threading

from hermes_cli import cli_shutdown


class _Agent:
    def __init__(self, review_seconds):
        self._session_messages = [{"role": "user", "content": "I use Fish, never bash"}]
        self._memory_manager = None
        self._background_review_run = None
        self.finished = threading.Event()
        self._review_seconds = review_seconds

    def shutdown_memory_provider(self, messages=None):
        run = self._background_review_run = type("Run", (), {"request_done": threading.Event()})()

        def review():
            self.finished.wait(self._review_seconds)
            self.finished.set()
            run.request_done.set()
        threading.Thread(target=review, daemon=True, name="bg-review").start()


def test_cli_exit_waits_for_the_session_end_review():
    agent = _Agent(review_seconds=0.3)
    cli_shutdown._shutdown_agent_memory_provider(agent)
    assert agent.finished.is_set()


def test_cli_exit_wait_is_bounded(monkeypatch):
    monkeypatch.setattr(cli_shutdown, "_SESSION_END_REVIEW_WAIT_S", 0.1)
    agent = _Agent(review_seconds=30)
    cli_shutdown._shutdown_agent_memory_provider(agent)
    assert not agent.finished.is_set()
