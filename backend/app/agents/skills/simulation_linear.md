Single-path simulation skill.

Goal:
Produce one likely next-step projection from the frozen world context.

Rules:
- Generate only one projection; do not create alternative scenarios or branches.
- Keep historical evidence separate from simulated events.
- Use no more than 5 outcomes, 6 public timeline items, and 5 watchpoints.
- Reuse known actors. Introduce a new actor only when it is necessary for a high-impact development.
- An agent may wait, monitor, or take no new action when that is most plausible.
- Treat each agent's bounded state as its decision context: beliefs interpret public state; desires express goals; capabilities and constraints bound conduct; decision policy shapes style; Timeline supplies memory.
- Treat supplied ordinary-agent proposals as the complete action set. Resolve them without rewriting or adding actor decisions.
- Own the complete semantic update of WorldState. Public facts go to WorldState and public_events.
- Do not modify BDI, Agent profiles, relationships, or the Registry during settlement.
- Return exactly one private-information list per known Agent. Do not explain or justify why an Agent knows a private fact.
- Keep the result compact enough for graph and timeline updates.
