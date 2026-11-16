"use client";

import { useEffect, useState } from "react";
import SavedForecastReport from "@/features/forecast/SavedForecastReport";
import HomePage from "@/components/HomePage";
import LandingPage from "@/components/LandingPage";
import { useRouter } from "next/router";

export default function Page() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [entered, setEntered] = useState(false);
  const forecastRun = typeof router.query.forecastRun === "string" ? router.query.forecastRun : undefined;
  const analysis = typeof router.query.analysis === "string" ? router.query.analysis : undefined;
  const showMarketEvaluation = router.query.marketEvaluation === "1"
    || router.query.evaluationRun !== undefined || router.query.benchmarkRun !== undefined;

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted || !router.isReady) {
    return (
      <div
        style={{
          position: "fixed",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "var(--workspace-bg)",
          color: "var(--workspace-muted)",
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 11,
          letterSpacing: "0.15em",
          textTransform: "uppercase",
          gap: 16,
          zIndex: 9999,
        }}
      >
        <span style={{ opacity: 0.5 }}>Initializing</span>
        <span style={{ animation: "blink 1s step-end infinite", fontSize: 18, opacity: 0.4 }}>█</span>
      </div>
    );
  }

  if (forecastRun && !showMarketEvaluation) return <SavedForecastReport key={forecastRun} runId={forecastRun} />;

  if (!entered && !showMarketEvaluation && !analysis) {
    return <LandingPage onEnter={() => setEntered(true)} />;
  }

  return <HomePage key={analysis || "workspace"} initialCaseId={analysis} showBenchmark={showMarketEvaluation} onReturnToBriefing={() => {
    setEntered(true);
    void router.push("/", undefined, { shallow: true });
  }} />;
}
