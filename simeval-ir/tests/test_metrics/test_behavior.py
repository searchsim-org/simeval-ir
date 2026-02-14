"""Tests for behavioral metrics."""

import pytest
import numpy as np

from simeval_ir.metrics.behavior.jsd import JSDActionTypes
from simeval_ir.metrics.behavior.distribution import (
    SessionLengthDistribution,
    TimingDistribution,
)


class TestJSDActionTypes:
    """Tests for JSD action types metric."""

    def test_compute(self, real_sessions, sim_sessions):
        """Test computing JSD between action type distributions."""
        metric = JSDActionTypes()
        result = metric.compute(real=real_sessions, sim=sim_sessions)

        assert result.name == "jsd_action_types"
        assert 0 <= result.value <= 1
        assert "action_types" in result.meta

    def test_identical_distributions(self, real_sessions):
        """Test that identical distributions have JSD ≈ 0."""
        metric = JSDActionTypes()
        result = metric.compute(real=real_sessions, sim=real_sessions)

        assert result.value < 0.01


class TestSessionLengthDistribution:
    """Tests for session length distribution metric."""

    def test_compute(self, real_sessions, sim_sessions):
        """Test computing KS statistic for session lengths."""
        metric = SessionLengthDistribution()
        result = metric.compute(real=real_sessions, sim=sim_sessions)

        assert result.name == "session_length_distribution"
        assert 0 <= result.value <= 1
        assert "p_value" in result.meta
        assert "mean_real" in result.meta
        assert "mean_sim" in result.meta


class TestTimingDistribution:
    """Tests for timing distribution metric."""

    def test_compute(self, real_sessions, sim_sessions):
        """Test computing KS statistic for timing distributions."""
        metric = TimingDistribution()
        result = metric.compute(real=real_sessions, sim=sim_sessions)

        assert result.name == "timing_distribution"
        # Value may be 0 if no timing data
        assert 0 <= result.value <= 1
