"""Tests for the Self-Learning Feedback System."""
from __future__ import annotations

import duckdb

from footy.self_learning import (
    PerformanceWindow,
    ExpertTracker,
    DriftDetector,
    SelfLearningLoop,
    get_learning_loop,
)


class TestPerformanceWindow:
    def test_empty_window(self):
        pw = PerformanceWindow()
        assert pw.accuracy == 0.0
        assert pw.n_predictions == 0
        assert pw.log_loss == 999.0
        assert pw.brier_score == 1.0

    def test_add_and_accuracy(self):
        pw = PerformanceWindow()
        pw.add({"correct": True, "predicted_prob": 0.7, "probs": [0.7, 0.2, 0.1], "outcome": 0})
        pw.add({"correct": False, "predicted_prob": 0.3, "probs": [0.3, 0.3, 0.4], "outcome": 2})
        assert pw.n_predictions == 2
        assert pw.accuracy == 0.5

    def test_max_size_enforcement(self):
        pw = PerformanceWindow(max_size=5)
        for _ in range(10):
            pw.add({"correct": True, "predicted_prob": 0.8})
        assert pw.n_predictions == 5

    def test_log_loss(self):
        pw = PerformanceWindow()
        pw.add({"correct": True, "predicted_prob": 0.9, "probs": [0.9, 0.05, 0.05], "outcome": 0})
        assert pw.log_loss < 0.2

    def test_brier_score(self):
        pw = PerformanceWindow()
        # Perfect prediction
        pw.add({"correct": True, "predicted_prob": 1.0, "probs": [1.0, 0.0, 0.0], "outcome": 0})
        assert pw.brier_score < 0.01


class TestExpertTracker:
    def test_default_weight(self):
        et = ExpertTracker(name="test")
        result = et.get_weight()
        # get_weight returns (weight, uncertainty) tuple
        assert result[0] == 1.0

    def test_record_updates_global(self):
        et = ExpertTracker(name="test")
        for i in range(60):
            et.record({
                "correct": i % 2 == 0,
                "predicted_prob": 0.6 if i % 2 == 0 else 0.4,
                "probs": [0.6, 0.2, 0.2],
                "outcome": 0,
                "log_loss": 0.5,
            })
        assert et.global_window.n_predictions == 60
        w, uncertainty = et.get_weight()
        assert w > 0

    def test_league_specific_weight(self):
        et = ExpertTracker(name="test")
        for _ in range(40):
            et.record(
                {"correct": True, "predicted_prob": 0.8, "probs": [0.8, 0.1, 0.1], "outcome": 0, "log_loss": 0.2},
                league="PL",
            )
        w_pl, _ = et.get_weight(league="PL")
        assert w_pl > 0


class TestDriftDetector:
    def test_no_drift_initially(self):
        dd = DriftDetector()
        assert not dd.is_drifting

    def test_no_drift_with_stable_values(self):
        dd = DriftDetector(min_instances=10)
        for _ in range(50):
            dd.update(0.5)
        assert not dd.is_drifting

    def test_drift_with_sudden_change(self):
        dd = DriftDetector(min_instances=10, lambda_threshold=20.0)
        # Stable period
        for _ in range(30):
            dd.update(0.5)
        # Sudden drift
        for _ in range(50):
            dd.update(3.0)
        assert dd.is_drifting

    def test_reset(self):
        dd = DriftDetector()
        for _ in range(50):
            dd.update(5.0)
        dd.reset()
        assert not dd.is_drifting


class TestSelfLearningLoop:
    def test_record_prediction(self):
        loop = SelfLearningLoop()
        result = loop.record_prediction_result(
            match_id=1,
            predicted_probs=[0.6, 0.2, 0.2],
            actual_outcome=0,
        )
        assert result["correct"] is True
        assert result["log_loss"] > 0
        assert result["n_predictions"] == 1

    def test_incorrect_prediction(self):
        loop = SelfLearningLoop()
        result = loop.record_prediction_result(
            match_id=1,
            predicted_probs=[0.6, 0.2, 0.2],
            actual_outcome=2,
        )
        assert result["correct"] is False

    def test_expert_weight_tracking(self):
        loop = SelfLearningLoop()
        for i in range(60):
            loop.record_prediction_result(
                match_id=i,
                predicted_probs=[0.5, 0.25, 0.25],
                actual_outcome=0 if i % 2 == 0 else 1,
                expert_predictions={
                    "elo": [0.6, 0.2, 0.2],
                    "form": [0.4, 0.3, 0.3],
                },
                league="PL",
            )
        weights = loop.get_optimal_expert_weights(league="PL")
        assert "elo" in weights
        assert "form" in weights
        assert sum(weights.values()) > 0.99

    def test_performance_report(self):
        loop = SelfLearningLoop()
        for i in range(20):
            loop.record_prediction_result(
                match_id=i,
                predicted_probs=[0.6, 0.2, 0.2],
                actual_outcome=0,
            )
        report = loop.get_performance_report()
        assert "overall" in report
        assert report["overall"]["n_predictions"] == 20
        assert report["overall"]["accuracy"] == 1.0

    def test_serialize(self):
        loop = SelfLearningLoop()
        loop.record_prediction_result(
            match_id=1,
            predicted_probs=[0.5, 0.3, 0.2],
            actual_outcome=0,
        )
        data = loop.serialize()
        assert "overall_accuracy" in data
        assert "overall_n" in data

    def test_from_db_reconstructs_expert_predictions(self):
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE matches (
                match_id BIGINT,
                competition VARCHAR,
                status VARCHAR,
                home_goals INT,
                away_goals INT,
                utc_date TIMESTAMP
            )
        """)
        con.execute("""
            CREATE TABLE predictions (
                match_id BIGINT,
                model_version VARCHAR,
                p_home DOUBLE,
                p_draw DOUBLE,
                p_away DOUBLE,
                notes VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE prediction_scores (
                match_id BIGINT,
                model_version VARCHAR,
                scored_at TIMESTAMP,
                outcome INT,
                correct BOOLEAN,
                logloss DOUBLE
            )
        """)
        con.execute("""
            CREATE TABLE expert_cache (
                match_id BIGINT,
                breakdown_json VARCHAR
            )
        """)

        con.execute("""
            INSERT INTO matches VALUES
            (1, 'PL', 'FINISHED', 2, 1, CURRENT_TIMESTAMP)
        """)
        con.execute("""
            INSERT INTO predictions VALUES
            (1, 'v10_council', 0.6, 0.25, 0.15, '{"btts":0.55}')
        """)
        con.execute("""
            INSERT INTO prediction_scores VALUES
            (1, 'v10_council', CURRENT_TIMESTAMP, 0, TRUE, 0.51)
        """)
        con.execute("""
            INSERT INTO expert_cache VALUES
            (1, '{"experts":{"elo":{"probs":{"home":0.62,"draw":0.20,"away":0.18},"confidence":0.71},"form":{"probs":{"home":0.58,"draw":0.24,"away":0.18},"confidence":0.66}}}')
        """)

        loop = SelfLearningLoop.from_db(con)
        weights = loop.get_optimal_expert_weights(league="PL")

        assert loop.overall_performance.n_predictions == 1
        assert "elo" in weights
        assert "form" in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        con.close()


class TestGetLearningLoop:
    def test_singleton(self):
        loop1 = get_learning_loop()
        loop2 = get_learning_loop()
        assert loop1 is loop2
