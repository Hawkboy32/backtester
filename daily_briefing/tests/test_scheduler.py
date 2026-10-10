from __future__ import annotations

from datetime import datetime, timezone

import scheduler


def test_should_run_now_is_false_before_the_cutoff_hour(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "STATE_PATH", tmp_path / "state.json")
    before_cutoff = datetime(2026, 10, 9, scheduler.CUTOFF_HOUR_UTC - 1, 0, tzinfo=timezone.utc)

    assert scheduler._should_run_now(before_cutoff) is False


def test_should_run_now_is_true_after_the_cutoff_with_no_prior_run(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "STATE_PATH", tmp_path / "state.json")
    after_cutoff = datetime(2026, 10, 9, scheduler.CUTOFF_HOUR_UTC, 1, tzinfo=timezone.utc)

    assert scheduler._should_run_now(after_cutoff) is True


def test_should_run_now_is_false_if_already_run_today(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "STATE_PATH", tmp_path / "state.json")
    scheduler._save_last_run_date("2026-10-09")
    after_cutoff = datetime(2026, 10, 9, scheduler.CUTOFF_HOUR_UTC, 30, tzinfo=timezone.utc)

    assert scheduler._should_run_now(after_cutoff) is False


def test_should_run_now_is_true_again_the_next_day(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "STATE_PATH", tmp_path / "state.json")
    scheduler._save_last_run_date("2026-10-09")
    next_day = datetime(2026, 10, 10, scheduler.CUTOFF_HOUR_UTC, 0, tzinfo=timezone.utc)

    assert scheduler._should_run_now(next_day) is True


def test_a_missing_state_file_is_treated_as_never_run(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "STATE_PATH", tmp_path / "does-not-exist.json")
    assert scheduler._load_last_run_date() is None


def test_an_unreadable_state_file_fails_safe_to_never_run(tmp_path, monkeypatch):
    bad_path = tmp_path / "state.json"
    bad_path.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(scheduler, "STATE_PATH", bad_path)

    assert scheduler._load_last_run_date() is None
