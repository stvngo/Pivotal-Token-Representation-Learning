"""Tests for the PTS search harness. No model, no GPU, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_pipeline.artifacts_io import iter_jsonl
from pts_harness.backends import ScriptedBackend
from pts_harness.checkpoint import RunStore
from pts_harness.probability import (
    ProbEstimate,
    delta_is_significant,
    false_accept_rate,
    node_key,
    node_seed,
    two_proportion_p,
)
from pts_harness.scheduler import QuerySpec, WaveScheduler
from pts_harness.search import Phase, QueryState, SearchConfig

PROMPT = [1, 2, 3]
GEN = list(range(100, 116))  # 16 generated tokens
JUMP = 7                      # probability jumps once token index 7 is included


def success_at_jump(prefix):
    return 0.8 if len(prefix) - len(PROMPT) > JUMP else 0.3


def flat(p):
    return lambda prefix: p


def oracle(_query, text):
    return "####42" in text


def make_backend(success_fn, gen=GEN):
    return ScriptedBackend(success_fn=success_fn, generation=lambda _p: gen)


def drive(qs: QueryState, backend, gen=GEN, max_waves=60):
    """Minimal single-query driver, mirroring what the scheduler does."""
    cache: dict[str, ProbEstimate] = {}
    events = []
    for _ in range(max_waves):
        if qs.is_done:
            break
        reqs = qs.ready_requests(cache)
        if reqs:
            for res in backend.generate(reqs):
                cache[res.request_id] = ProbEstimate(
                    res.request_id,
                    len(res.rollouts),
                    sum(oracle("", r.text) for r in res.rollouts),
                )
        elif qs.needs_generation():
            qs.supply_generation(gen)
            continue
        events.extend(qs.advance(cache))
    return events, cache


# -- probability -----------------------------------------------------------


def test_wilson_interval_brackets_the_estimate():
    est = ProbEstimate("k", 50, 25)
    lo, hi = est.wilson_ci()
    assert lo < est.p < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_is_well_behaved_at_the_boundary():
    lo, hi = ProbEstimate("k", 40, 0).wilson_ci()
    assert lo == 0.0 and 0.0 < hi < 0.2


def test_false_accept_rate_matches_the_reported_label_noise():
    """PTS defaults (S=50, tau=0.2) put the threshold at only 2 sigma."""
    assert false_accept_rate(50, 0.2) == pytest.approx(0.0455, abs=5e-4)
    assert false_accept_rate(100, 0.2) < false_accept_rate(40, 0.2)


def test_significance_gate_is_stricter_than_the_threshold_alone():
    before, after = ProbEstimate("a", 20, 8), ProbEstimate("b", 20, 14)
    assert delta_is_significant(before, after, 0.2, alpha=None)
    assert not delta_is_significant(before, after, 0.2, alpha=0.01)
    assert two_proportion_p(before, after) > 0.01


def test_node_keys_are_content_addressed():
    """Two paths reaching the same prefix must share the estimate -- that is
    what makes a bisection node cost one new estimate rather than two."""
    assert node_key("m", "q", [1, 2, 3]) == node_key("m", "q", (1, 2, 3))
    assert node_key("m", "q", [1, 2, 3]) != node_key("m", "q", [1, 2, 4])
    assert node_key("m", "q1", [1]) != node_key("m", "q2", [1])


def test_node_seed_is_stable_across_batch_composition():
    a = node_seed(42, "q", "key")
    assert a == node_seed(42, "q", "key")
    assert a != node_seed(43, "q", "key")


# -- the search itself -----------------------------------------------------


def test_finds_the_planted_pivot_at_the_exact_position():
    qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1))
    events, _ = drive(qs, make_backend(success_at_jump))

    assert len(events) == 1
    ev = events[0]
    assert ev.position == len(PROMPT) + JUMP, "absolute, prompt-inclusive index"
    assert ev.token_id == GEN[JUMP]
    assert ev.prob_delta == pytest.approx(0.5, abs=1e-6)
    assert ev.is_positive


def test_prefix_ids_stop_before_the_pivot():
    """prefix_token_ids is the t-1 context: everything up to but excluding
    the pivotal token. This is what the probe pipeline reads."""
    qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1))
    events, _ = drive(qs, make_backend(success_at_jump))
    ev = events[0]
    assert ev.prefix_token_ids == PROMPT + GEN[:JUMP]
    assert len(ev.prefix_token_ids) == ev.position


def test_no_events_when_probability_never_moves():
    qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1))
    events, _ = drive(qs, make_backend(flat(0.5)))
    assert events == []


def test_query_outside_the_probability_band_is_skipped():
    """PTS only searches mid-difficulty questions: a task the model always or
    never solves has no pivotal token, because a saturated probability
    cannot move."""
    for p in (0.05, 0.95):
        qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1))
        events, _ = drive(qs, make_backend(flat(p)))
        assert events == []
        assert qs.skipped
        assert qs.phase is Phase.DONE


def test_short_generations_are_skipped():
    qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1, min_generation_len=5))
    events, _ = drive(qs, make_backend(success_at_jump, gen=[1, 2]), gen=[1, 2])
    assert events == []


def test_each_node_costs_one_new_estimate():
    """The cache turns a node's endpoints into parent hits, so only the
    midpoint is new. This is the basis of the cost model."""
    qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1))
    _, cache = drive(qs, make_backend(success_at_jump))
    assert len(cache) <= qs.n_nodes + 1, (
        f"{len(cache)} estimates for {qs.n_nodes} nodes; caching is not working"
    )


def test_two_pivots_are_both_found():
    # Baseline must sit inside [min_prob, max_prob] or the query is screened
    # out before any search happens.
    def two_jumps(prefix):
        g = len(prefix) - len(PROMPT)
        if g > 11:
            return 0.85
        if g > 3:
            return 0.55
        return 0.3

    qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1))
    events, _ = drive(qs, make_backend(two_jumps))
    positions = sorted(e.position - len(PROMPT) for e in events)
    assert positions == [3, 11]


def test_negative_pivot_is_recorded_with_its_sign():
    def drops(prefix):
        return 0.2 if len(prefix) - len(PROMPT) > JUMP else 0.7

    qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1))
    events, _ = drive(qs, make_backend(drops))
    assert len(events) == 1
    assert events[0].prob_delta < 0
    assert not events[0].is_positive


def test_event_carries_sufficient_statistics_not_a_verdict():
    """Storing (n, successes) rather than p_hat keeps prob_threshold post-hoc
    and lets every event carry a confidence interval."""
    qs = QueryState("q1", PROMPT, SearchConfig(max_generations=1, num_samples=40))
    events, _ = drive(qs, make_backend(success_at_jump))
    d = events[0].to_dict()
    assert d["n_before"] == 40 and d["n_after"] == 40
    assert d["s_before"] == 12 and d["s_after"] == 32
    assert len(d["ci_before"]) == 2 and d["ci_before"][0] < d["prob_before"] < d["ci_before"][1]


# -- scheduler -------------------------------------------------------------


def test_batching_across_queries_does_not_add_waves():
    """Wall clock should scale with tree DEPTH, not with the number of
    queries -- that is the entire reason for the wave scheduler."""
    cfg = SearchConfig(max_generations=1)

    def waves_for(n):
        backend = make_backend(success_at_jump)
        sched = WaveScheduler(backend, oracle, cfg, max_active=n)
        specs = [QuerySpec(f"q{i}", PROMPT, query=f"q{i}") for i in range(n)]
        return sched.run(specs)

    one, many = waves_for(1), waves_for(16)
    assert many.events == 16
    assert many.waves == one.waves, f"{many.waves} waves for 16 queries vs {one.waves} for 1"
    assert many.rollouts > one.rollouts


def test_scheduler_finds_one_event_per_query():
    cfg = SearchConfig(max_generations=1)
    sched = WaveScheduler(make_backend(success_at_jump), oracle, cfg, max_active=8)
    summary = sched.run([QuerySpec(f"q{i}", PROMPT) for i in range(8)])
    assert summary.queries_completed == 8
    assert summary.events == 8
    assert summary.queries_skipped == 0


def test_scheduler_counts_skipped_queries():
    cfg = SearchConfig(max_generations=1)
    sched = WaveScheduler(make_backend(flat(0.95)), oracle, cfg)
    summary = sched.run([QuerySpec(f"q{i}", PROMPT) for i in range(4)])
    assert summary.queries_skipped == 4
    assert summary.events == 0


# -- checkpoint and resume -------------------------------------------------


def test_resume_recomputes_nothing(tmp_path: Path):
    """The point of the checkpoint: a killed session restarts a half-done
    query from the top, but every node it revisits is a cache hit."""
    cfg = SearchConfig(max_generations=1)
    specs = [QuerySpec(f"q{i}", PROMPT, query=f"q{i}") for i in range(6)]

    b1 = make_backend(success_at_jump)
    with RunStore(tmp_path / "run", session="s1") as store:
        first = WaveScheduler(b1, oracle, cfg, store=store).run(specs)
    assert first.events == 6

    b2 = make_backend(success_at_jump)
    with RunStore(tmp_path / "run", session="s2") as store:
        second = WaveScheduler(b2, oracle, cfg, store=store).run(specs)

    assert second.resumed_from_cache > 0
    assert b2.n_rollouts == 0, "resume issued rollouts for already-finished queries"
    assert second.events == 0


def test_resume_completes_queries_the_first_pass_never_saw(tmp_path: Path):
    cfg = SearchConfig(max_generations=1)
    early = [QuerySpec(f"q{i}", PROMPT, query=f"q{i}") for i in range(3)]
    everything = [QuerySpec(f"q{i}", PROMPT, query=f"q{i}") for i in range(6)]

    with RunStore(tmp_path / "run", session="s1") as store:
        WaveScheduler(make_backend(success_at_jump), oracle, cfg, store=store).run(early)
    with RunStore(tmp_path / "run", session="s2") as store:
        second = WaveScheduler(
            make_backend(success_at_jump), oracle, cfg, store=store
        ).run(everything)

    assert second.events == 3, "only the three new queries should be searched"
    with RunStore(tmp_path / "run", session="s3") as store:
        assert len(store.load_events()) == 6
        assert len(store.completed_ids()) == 6


def test_events_are_deduped_across_sessions(tmp_path: Path):
    store = RunStore(tmp_path / "run", session="s1")
    with store:
        store.record_events([{"query_uid": "q", "position": 5, "prefix_len": 5}] * 3)
    assert len(RunStore(tmp_path / "run", session="s2").load_events()) == 1


def test_a_torn_final_line_does_not_lose_the_run(tmp_path: Path):
    """A session killed mid-write leaves a partial last line; the run must
    still resume from everything before it."""
    cfg = SearchConfig(max_generations=1)
    with RunStore(tmp_path / "run", session="s1") as store:
        WaveScheduler(make_backend(success_at_jump), oracle, cfg, store=store).run(
            [QuerySpec("q0", PROMPT, query="q0")]
        )

    probs = tmp_path / "run" / "sessions" / "s1" / "probs.jsonl"
    n_before = len(list(iter_jsonl(probs)))
    with probs.open("a") as fh:
        fh.write('{"k": "partial", "n": 4')

    cache = RunStore(tmp_path / "run", session="s2").load_prob_cache()
    assert len(cache) == n_before


def test_heartbeat_is_written_atomically(tmp_path: Path):
    cfg = SearchConfig(max_generations=1)
    with RunStore(tmp_path / "run", session="s1") as store:
        WaveScheduler(make_backend(success_at_jump), oracle, cfg, store=store).run(
            [QuerySpec("q0", PROMPT, query="q0")]
        )
        hb = json.loads(store.paths.heartbeat.read_text())
    assert "wave" in hb and "elapsed_s" in hb


def test_config_is_recorded_once(tmp_path: Path):
    RunStore(tmp_path / "run", session="s1", config={"num_samples": 40})
    RunStore(tmp_path / "run", session="s2", config={"num_samples": 99})
    cfg = json.loads((tmp_path / "run" / "config.json").read_text())
    assert cfg["num_samples"] == 40, "the first session's config is authoritative"
