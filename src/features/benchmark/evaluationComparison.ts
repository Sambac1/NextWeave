import type { BenchmarkMethod } from "@/services/benchmarkService";
import type { EvaluationMethod, EvaluationRun } from "./EvaluationResults";

export const comparisonMethods = [
  ["direct_qa", "Direct QA", "direct_qa"],
  ["qa_ensemble", "QA Ensemble", "qa_ensemble"],
  ["single_trajectory_simulation", "Single-trajectory Simulation", "single_multistep"],
  ["mcts_simulation", "MCTS Simulation", "mcts_simulation"],
  ["multi_trajectory_simulation", "NextWeave", "trajectory_sampling"],
] as const satisfies readonly (readonly [EvaluationMethod, string, BenchmarkMethod])[];

export const comparisonRunLabels = Object.fromEntries(comparisonMethods.map(([, label, method]) => [method, label]));

type Question = NonNullable<EvaluationRun["questions"]>[number];
type Mode = EvaluationRun["mode"];

export function isProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

export function getReference(question: Question, mode: Mode) {
  if (mode === "live") return isProbability(question.quote) ? question.quote : null;
  return question.resolution === "Resolved" && typeof question.observed_yes === "boolean"
    ? Number(question.observed_yes)
    : null;
}

export function scoreEvaluation(questions: { question: Question; reference: number }[], method: EvaluationMethod, mode: Mode) {
  let squaredError = 0;
  let secondaryError = 0;
  for (const { question, reference } of questions) {
    const forecast = question.forecasts[method];
    const probability = isProbability(forecast) ? forecast : 0.5;
    squaredError += (probability - reference) ** 2;
    if (mode === "historical") {
      const clipped = Math.max(1e-15, Math.min(1 - 1e-15, probability));
      secondaryError -= reference ? Math.log(clipped) : Math.log1p(-clipped);
    } else {
      secondaryError += Math.abs(probability - reference);
    }
  }

  const total = questions.length;
  if (!total) return { primary: null, secondary: null };
  return mode === "historical"
    ? { primary: squaredError / total, secondary: secondaryError / total }
    : { primary: (secondaryError / total) * 100, secondary: Math.sqrt(squaredError / total) * 100 };
}
