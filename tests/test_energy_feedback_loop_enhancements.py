import time

from model.energy_feedback_loop import (
    ControlAction,
    ControlConfig,
    EnergyFeedbackConfig,
    EnergyFeedbackController,
    EnergyMetrics,
)


def make_metrics(power, latency, throughput, memory, util, batch=4, worker_id="worker-0"):
    return EnergyMetrics(
        power_w=power,
        latency_ms=latency,
        throughput_tokens_per_sec=throughput,
        memory_used_mb=memory,
        memory_total_mb=memory * 2,
        gpu_utilization_pct=util,
        batch_size=batch,
        worker_id=worker_id,
    )


def test_controller_accepts_legacy_and_config_api():
    config = EnergyFeedbackConfig(power_target_w=120.0, latency_target_ms=40.0)
    controller = EnergyFeedbackController(config=config)
    assert controller.get_state()["targets"]["power_w"] == 120.0

    legacy = EnergyFeedbackController(power_target_w=130.0, latency_target_ms=45.0)
    assert legacy.get_state()["targets"]["power_w"] == 130.0


def test_predictive_control_reacts_to_rising_power():
    controller = EnergyFeedbackController(
        EnergyFeedbackConfig(
            power_target_w=100.0,
            latency_target_ms=60.0,
            update_interval_s=1.0,
            smoothing_window=3,
            stability_window=3,
        )
    )

    # Build a rising trend to trigger proactive power reduction.
    controller.record_metrics(make_metrics(90, 50, 28, 2000, 50))
    controller.record_metrics(make_metrics(98, 52, 27, 2000, 55))
    controller.record_metrics(make_metrics(110, 54, 26, 2000, 58))

    decision = controller._select_best_action(make_metrics(115, 55, 25, 2000, 60))

    assert decision.action in {
        ControlAction.DECREASE_POWER_LIMIT,
        ControlAction.DECREASE_BATCH,
    }
    assert decision.score != 0


def test_stability_detection_uses_variance_window():
    config = EnergyFeedbackConfig(stability_window=4, stability_variance_power=2.0, stability_variance_latency=2.0)
    controller = EnergyFeedbackController(config=config)

    for sample in [
        make_metrics(100.0, 50.0, 25.0, 1800, 50),
        make_metrics(101.0, 49.0, 25.2, 1810, 51),
        make_metrics(99.5, 50.5, 25.1, 1790, 50),
        make_metrics(100.5, 49.5, 25.0, 1805, 50),
    ]:
        controller.record_metrics(sample)

    controller._update_stability(make_metrics(100.0, 50.0, 25.0, 1800, 50))
    controller._state.stability_since = time.monotonic() - 25.0

    assert controller._is_stable() is True


def test_action_scoring_prefers_low_risk_batch_tuning():
    controller = EnergyFeedbackController(
        EnergyFeedbackConfig(
            power_target_w=180.0,
            latency_target_ms=120.0,
            stability_window=3,
            update_interval_s=1.0,
        )
    )

    stable_metrics = make_metrics(110, 70, 35, 1500, 42)
    for _ in range(3):
        controller.record_metrics(stable_metrics)
    controller._update_stability(stable_metrics)
    controller._state.stability_since = time.monotonic() - 30.0

    decision = controller._select_best_action(stable_metrics)

    assert decision.action == ControlAction.INCREASE_BATCH
    assert decision.risk >= 0.0


def test_safe_step_returns_config_and_uses_aliases():
    controller = EnergyFeedbackController()
    config = controller.step()

    assert isinstance(config, ControlConfig)
    assert config.power_limit_w == config.gpu_power_limit_w
    assert config.offload_threshold == config.offload_threshold_mb
