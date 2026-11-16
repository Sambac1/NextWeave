# OKF and trajectory simulation

## Traceable knowledge

A confirmed forecast freezes its question, evidence cutoff, and resolution criteria. Collection captures Wikipedia revisions available at the cutoff. Source body text is stored once; a separate source index records revision metadata, file locations, and fingerprints. Sections provide bounded extraction and retrieval units without duplicating the entire source.

Timeline extraction retains links to the passages that support an event and its date. Occurrence dates are distinct from scheduled dates, commitments, and contextual references. Material that cannot support a dated event may remain as background when permitted by its evidence policy; it does not acquire an invented date.

An OKF release is the public, versioned knowledge projection under `DATA_DIR/knowledge/`. Its documents link timeline events to participating actors and source snapshots. Actor documents carry supported identity, decision scope, and evidence references. Qualified decision actors receive a separate compact operational dossier and opening BDI state for simulation. Search indexes can be rebuilt from the knowledge documents. Run checkpoints and private observations remain outside the public knowledge release.

## Retrieval and decisions

A round starts from its committed world state. Context combines shared background, the actor's dossier, relevant historical events and passages, and observations visible to that actor. Retrieval is bounded so that repeated source material does not crowd out the current situation.

Activation determines which actors need a decision. Each active actor proposes up to three substantive alternatives, including waiting or observing when appropriate. Candidate review checks decision authority, feasibility, prerequisites, evidence visibility, and duplication. The eligible set can contain fewer than three actions.

For eligible actions `A(i,t)`, nonnegative weights are normalized into `p(i,t,a) = w(i,t,a) / sum(w(i,t,*))`. One action is sampled from that categorical distribution. This probability describes the actor's choice in its current simulated state; it does not establish that the attempted action succeeds.

## Joint settlement and state updates

The environment receives the round's selected actions together and resolves their interactions against the current world state and clock. It records attempted acts separately from execution, acceptance, and realized consequences. Settlement produces public developments, issue-state changes, and recipient-specific private observations. The simulator checks identity, time, and state constraints before committing the round.

Public developments enter the next round's shared context. Private observations enter only their recipients' states. Actors may revise their beliefs, assess progress toward desires, and adjust their current intentions. Candidate pools, review outcomes, normalized probabilities, random draws, and settlement records remain inspectable in the run audit.

## Probability aggregation

The main question workflow records three distinct quantities:

| Quantity | Meaning |
| --- | --- |
| Action probability `p(i,t,a)` | Conditional propensity for one eligible actor choice in one state. |
| Path likelihood | Product of the conditional probabilities of sampled choices, retained for audit. |
| Represented mass `m(k)` | Probability mass represented by trajectory `k` under the root sampling plan. |

When a material initial choice is stratified, each stratum's probability mass is divided among its allocated samples. Otherwise trajectories receive equal represented mass. The complete set of planned masses sums to one. Later actor decisions are sampled conditionally within each trajectory; multiplying path likelihood into represented mass again would double-count that sampling.

At the deadline, evidence checks first test whether settled simulated observations establish a definite Yes or No under the exact criteria. If neither is established, a conditional readout estimates `q(k)`, the Yes probability supported by the completed simulated path. It must respect the deadline and all question qualifiers. A proposal does not prove implementation, and a terminal readout must not invent further developments after the deadline.

For a fully assessed run, `P(Yes) = sum(m(k) * q(k))`. Each `q(k)` is in `[0,1]`; a verified result can set it to zero or one. This is a weighted conditional estimate, not simply the share of trajectories labeled Yes.

If some planned mass has no valid readout, let `Y = sum(m(k) * q(k))` over assessed trajectories and `U` be the unassessed mass. The report retains the range `[Y, Y + U]` and can show a provisional estimate conditional on assessed mass. It does not silently count missing results as No or label that provisional value a complete estimate.

These estimates depend on the frozen evidence, model, and configured settlement rules. Historical scoring against held-out official outcomes measures predictive performance; agreement with a live market snapshot measures a different reference.

## Report and trajectory replay

The report is assembled from saved public records. Replay displays each recorded public candidate and its original probability, highlights the selected action, and connects it to the settlement and subsequent round. Private options may leave publicly visible probabilities below 100%; the view does not renormalize them.

Planned and recorded rounds, interruption reasons, and incomplete trajectories remain explicit. The replay does not regenerate missing steps or alter saved estimates. Fixed Market Evaluation snapshots retain the method definitions and values stored with those runs.
