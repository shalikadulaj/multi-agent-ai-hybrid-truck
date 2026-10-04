"""Command-line entry point for the assumed P4 hybrid-truck EMS experiment."""

from __future__ import annotations

from .rl_training import main as run_p4_training


def main() -> None:
    """Train/evaluate the P4 Q-learning EMS and write report artifacts."""
    run_p4_training()


if __name__ == "__main__":
    main()
