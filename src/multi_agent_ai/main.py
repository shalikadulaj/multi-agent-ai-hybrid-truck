from __future__ import annotations

from .agents import HybridTruckSimulation, performance_visualization_agent


def main() -> None:
    sim = HybridTruckSimulation(steps=3)
    history = sim.run()

    print("\n=== Multi-Agent EMS Demo ===")
    for step in history:
        perf = step["agent_outputs"]["performance"]
        final_decision = step.get("final_decision")
        if final_decision is None:
            continue
        print(
            f"Step {step['timestamp']}: "
            f"gear={final_decision['selected_gear']}, "
            f"split={final_decision['torque_split']}, "
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
        print(f"\nGraph saved to: {plot_result['image_path']}")
    else:
        print("\nGraph not generated because matplotlib is not installed.")


if __name__ == "__main__":
    main()
