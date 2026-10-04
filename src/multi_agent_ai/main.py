from __future__ import annotations

from .agents import (
    HybridTruckSimulation,
    benchmark_visualization_agent,
    compare_policies,
    performance_visualization_agent,
)


def main() -> None:
    sim = HybridTruckSimulation(steps=40)
    history = sim.run()

    print("\n=== Realistic Multi-Agent Hybrid Truck EMS ===")
    for step in history[:10]:
        perf = step["agent_outputs"]["performance"]
        final_decision = step.get("final_decision")
        if final_decision is None:
            continue
        print(
            f"Step {step['timestamp']}: "
            f"driver_pedal={final_decision['driver_pedal']:.2f}, "
            f"engine={final_decision['engine_torque_nm']:.0f}Nm, "
            f"motor={final_decision['motor_torque_nm']:.0f}Nm, "
            f"regen={final_decision['regen_kw']:.1f}kW, "
            f"battery={step['battery_state']['soc_pct']:.1f}%, "
            f"reward={perf['total_reward']:.3f}, "
            f"status={perf['status']}"
        )

    final = history[-1]
    final_decision = final.get("final_decision")
    print("\nFinal decision:")
    print(final_decision)
    print("Performance:")
    print(final["agent_outputs"]["performance"])

    plot_result = performance_visualization_agent(history)
    if plot_result["status"] == "generated":
        print(f"\nTrajectory graph saved to: {plot_result['image_path']}")
    else:
        print("\nTrajectory graph not generated because matplotlib is not installed.")

    benchmark_result = compare_policies()
    benchmark_plot = benchmark_visualization_agent(benchmark_result)
    if benchmark_plot["status"] == "generated":
        print(f"Benchmark graph saved to: {benchmark_plot['image_path']}")
    else:
        print("Benchmark graph not generated because matplotlib is not installed.")
    print("\nBenchmark summary:")
    for policy_name, scores in benchmark_result.items():
        print(policy_name, scores)


if __name__ == "__main__":
    main()
