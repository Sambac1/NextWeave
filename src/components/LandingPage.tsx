import Image from "next/image";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { marketEvaluationPath } from "@/config/evaluationRoutes";
import styles from "./LandingPage.module.css";

const workflow = [
  { title: "Gather evidence", description: "Save relevant Wikipedia revisions and build a timeline." },
  { title: "Model decision actors", description: "Build profiles of the people and institutions that can act." },
  { title: "Compare simulated outcomes", description: "Sample actions over time and review the resulting forecast." },
];

const decisionX = 454;
const decisionY = 124;
const sampleRoutes = [
  { endX: 371, endY: 126, curve: "H204C262 124 306 128 339 127S360 126.25 371 126" },
  { endX: 350, endY: 92, curve: "H142C185 124 211 92 251 85C286 79 321 83 350 92" },
  { endX: 382, endY: 151, curve: "H120C164 124 206 151 252 159C301 167 350 163 382 151" },
  { endX: 324, endY: 61, curve: "H103C146 124 183 75 228 54C260 39 293 46 324 61" },
  { endX: 363, endY: 180, curve: "H92C135 124 168 161 214 181C266 203 334 198 363 180" },
];
const forecastCycleMs = 6600;

function BranchingForecast() {
  const figureRef = useRef<HTMLElement>(null);
  const [cycle, setCycle] = useState<number | null>(null);

  useEffect(() => {
    const figure = figureRef.current;
    if (!figure) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let inView = false;
    let interval: number | undefined;
    const playNextCycle = () => setCycle(current => (current ?? 0) + 1);
    const syncLoop = () => {
      if (!inView || document.hidden || reducedMotion.matches) {
        window.clearInterval(interval);
        interval = undefined;
        setCycle(null);
        return;
      }
      if (interval !== undefined) return;
      playNextCycle();
      interval = window.setInterval(playNextCycle, forecastCycleMs);
    };
    const observer = new IntersectionObserver(([entry]) => {
      inView = entry.intersectionRatio >= .2;
      syncLoop();
    }, { threshold: .2 });

    observer.observe(figure);
    document.addEventListener("visibilitychange", syncLoop);
    reducedMotion.addEventListener("change", syncLoop);
    return () => {
      window.clearInterval(interval);
      observer.disconnect();
      document.removeEventListener("visibilitychange", syncLoop);
      reducedMotion.removeEventListener("change", syncLoop);
    };
  }, []);

  return (
    <figure ref={figureRef} className={styles.figure} aria-label="Five illustrative forecast paths converge into one outcome distribution">
      <svg className={styles.diagram} viewBox="0 0 540 235" aria-hidden="true" focusable="false">
        <g key={cycle ?? "still"} className={styles.samples} data-playing={cycle !== null ? "" : undefined}>
          {sampleRoutes.map(({ endX, endY, curve }, index) => (
            <g key={endY} style={{
              "--draw-at": `${270 + index * 500}ms`,
              "--collect-at": `${2860 + index * 28}ms`,
              "--result-x": `${decisionX - endX}px`,
              "--result-y": `${decisionY - endY}px`,
            } as CSSProperties}>
              <path className={styles.route} d={`M64 124${curve}`} pathLength="1" />
              <path className={styles.collectTrail} d={`M${endX} ${endY}L${decisionX} ${decisionY}`} pathLength="1" />
              <g transform={`translate(${endX} ${endY})`}>
                <circle className={styles.sample} r="3.5" />
              </g>
            </g>
          ))}
          <g transform={`translate(${decisionX} ${decisionY})`}>
            <circle className={styles.decisionNode} r="6" />
            <g className={styles.resultBar}>
              <path className={styles.resultPrimary} d="M7 -6H-34a6 6 0 0 0 0 12H7Z" />
              <path className={styles.resultSecondary} d="M9 -6H34a6 6 0 0 1 0 12H9Z" />
            </g>
          </g>
        </g>
        <circle className={styles.origin} cx="64" cy="124" r="5" />
      </svg>
      <figcaption className={styles.stages}>
        <span>Start</span>
        <span>Simulation</span>
        <span>Decision</span>
      </figcaption>
    </figure>
  );
}

export default function LandingPage({ onEnter }: { onEnter: () => void }) {
  return (
    <main className={styles.page}>
      <div className={styles.frame}>
        <header className={styles.header}>
          <div className={styles.wordmark}>
            <Image src="/brand/nextweave-icon.png" alt="" width={44} height={44} priority />
            <span>NextWeave</span>
          </div>
          <Link href={marketEvaluationPath} shallow className={styles.evaluation}>Market Evaluation</Link>
        </header>
        <section className={styles.hero} aria-labelledby="landing-title">
          <div className={styles.intro}>
            <h1 id="landing-title">Forecast how an event could unfold</h1>
            <p>Start with a question or a Polymarket contract. NextWeave uses Wikipedia evidence to model who can act and estimate the chance of an outcome across simulated scenarios.</p>
            <button type="button" className={styles.primary} onClick={onEnter}>
              Start a forecast <ArrowRight size={18} aria-hidden="true" />
            </button>
          </div>
          <BranchingForecast />
        </section>
        <section aria-label="Forecast workflow">
          <ol className={styles.workflow}>
            {workflow.map(({ title, description }, index) => (
              <li key={title}>
                <span className={styles.step} aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
                <div><h2>{title}</h2><p>{description}</p></div>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </main>
  );
}
