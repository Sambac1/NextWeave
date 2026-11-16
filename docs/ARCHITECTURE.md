# Architecture

NextWeave has a statically exported Next.js frontend and a FastAPI backend. The backend owns evidence collection, model calls, actor preparation, simulation, and persistent checkpoints. The frontend submits tasks and displays their saved artifacts.

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Question workflow | `backend/app/api/events.py`, `backend/app/application/forecast_task.py` | Normalize a question or contract URL, confirm its cutoff, deadline, and resolution criteria, and discover relevant Wikipedia pages. |
| Historical knowledge | `backend/app/knowledge/` | Acquire source revisions, extract and validate events, preserve provenance, and publish OKF documents. |
| Actor preparation | `backend/app/knowledge/decision_seats.py`, `backend/app/knowledge/profile_evidence.py`, `backend/app/layers/pipeline/` | Identify decision owners, bind supporting passages, and prepare grounded profiles and opening BDI states. |
| Simulation | `backend/app/simulation/` | Retrieve context, activate actors, review and sample actions, settle joint consequences, and commit rounds. |
| Forecast report | `backend/app/simulation/question_report.py`, `src/features/forecast/` | Assemble saved public results and render the estimate, evidence, and trajectory replay. |
| Market evaluation | `backend/app/benchmark/`, `src/features/benchmark/` | Import contracts, freeze evaluation inputs, run forecasting methods, and score or inspect their results. |
| Storage | `backend/app/storage/` | Persist source bodies, indexes, knowledge releases, task state, and reports. |

## Evidence and actor preparation

A confirmed question establishes the evidence cutoff separately from the forecast deadline. Wikipedia discovery ranks relevant article titles and retrieves eligible revisions. Stored source bodies and indexes preserve revision identifiers, timestamps, source paths, and fingerprints. Timeline events cite exact source passages; temporal validation distinguishes occurrences from proposals, schedules, and contextual dates.

Actor preparation identifies the units that can make decisions relevant to the question. A principal and its representative must not independently own the same choice. Identity evidence and decision evidence may come from separate passages, but each must bind to the same actor. Canonical identifiers preserve links between aliases, historical events, and profiles.

A qualified actor receives a compact dossier covering its supported identity, decision scope, authority, resources, constraints, and BDI. Evidence-backed statements and modeling assumptions remain distinguishable. Unknown fields are not filled with invented facts. Required actors and their decision states are checked before preparation is marked complete. Incomplete preparation retains successful timeline and extraction artifacts for a bounded retry.

## Simulation lifecycle

Each run freezes its question, knowledge release, actor inputs, and clock. A round reads the previously committed state. Actors receive public background, relevant retrieved evidence, and their own private observations. Activation determines which actors need to deliberate. Active actors propose up to three distinct options; review checks actor scope, prerequisites, visibility, and duplication before sampling.

All selected actions are settled together. The settlement separates attempted acts, executed decisions, observable consequences, and private information. Time and state constraints are checked before the updated world state, public events, and recipient-specific observations are committed. The next round reads that committed state. Cached stages and round checkpoints support recovery without silently presenting an incomplete run as complete.

The main workflow stratifies a material initial decision when available. A trajectory's represented sampling mass is separate from its sampled path likelihood. At the deadline, supported simulated observations can establish a definite result; otherwise a conditional readout evaluates the exact target against the settled path. The report aggregates supported estimates with represented mass and retains unresolved mass. See [OKF and trajectory simulation](OKF_AND_ENSEMBLE.md) for the calculation.

## Persistence boundaries

- **Source store:** captured source bodies, revision metadata, and section indexes.
- **Public OKF release:** immutable, versioned documents linking events, actors, and original evidence.
- **Forecast state:** frozen inputs, candidate pools, audits, sampled decisions, committed rounds, and private observations.
- **Report projection:** public simulation records and their probability readout; report assembly does not rerun the model.
- **Working evaluations:** active contract imports, method runs, and settlement refreshes under `DATA_DIR/benchmark/`.
- **Bundled comparison:** fixed results under `backend/evaluation_data/historical-live/`, independent of runtime analysis cleanup.

The comparison endpoints `/api/benchmark/comparison/evaluations/{id}` and `/api/benchmark/comparison/runs/{id}` read the bundled archive without requiring working evaluation data. Its manifest records SHA-256 checksums. Runtime settlement refreshes do not rewrite these fixed snapshots.

## Frontend delivery

`npm run build` produces the static site in `out/`. The frontend backend URL is configured at build time through `NEXT_PUBLIC_API_URL`, with a local default for development. Provider credentials belong in backend configuration or request-level settings, never in public build variables. The backend controls allowed frontend origins through `FRONTEND_ORIGINS`.
