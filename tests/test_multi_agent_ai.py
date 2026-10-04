from multi_agent_ai.agents import MultiAgentSystem, build_sample_state

system = MultiAgentSystem()
result = system.run(build_sample_state())

print(result["agent_outputs"]["performance"])
