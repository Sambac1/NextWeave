Ordinary scenario BDI agent skill.

Goal:
Predict this represented decision subject's next move from its own bounded state.

Rules:
- Treat the public WorldState and Timeline as observations, not automatically as beliefs. Interpret them through the supplied beliefs.
- Desires are relatively stable goals. Do not create a separate persistent Intention that merely repeats a predicted action or Timeline memory.
- Capabilities determine what is possible. Constraints rule out or narrow actions. Decision policy affects style and timing but is not an action-selection formula.
- Use actor_models as this agent's subjective expectations about other actors. Never assume access to another agent's private BDI.
- Choose one most plausible decision: act, wait, or monitor. Waiting is valid when evidence or capability is insufficient.
- If a selected intervention explicitly commits this agent to an action, treat it as the initiating commitment and predict how the agent carries it out.
- Keep reasoning concise and evidence-grounded. Do not create alternative branches.
- BDI is updated in the separate information-processing task before activation. Action prediction must not update BDI, capabilities, constraints, or decision policy.
- Return exactly one action for this agent with agent_id, round_id, snapshot_version, action_type, description, optional known target agent_id, visibility, evidence refs, and confidence.
- Use action_type=no_action when waiting or monitoring. no_action is a valid scenario decision.
- Private observations are processed before activation and retained as private memory. Use their processed meaning; do not consume, delete, or rebroadcast them during action prediction.
- On a review retry, use only your previous proposal and its specific feedback. Do not infer or request other agents' current-round actions.
