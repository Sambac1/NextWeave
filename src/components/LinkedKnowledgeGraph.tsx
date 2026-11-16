"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import cytoscape from "cytoscape";
import dagre from "cytoscape-dagre";
import {
  Building2,
  CircleDot,
  FileText,
  Flag,
  GitBranch,
  Maximize2,
  Minimize2,
  Minus,
  Plus,
  Scan,
  UserRound,
  Package,
  MapPin,
  Users,
} from "lucide-react";

cytoscape.use(dagre);
const NODE_WIDTH = 264;
const NODE_HEIGHT = 128;
type Graph = {
  nodes: any[];
  edges: any[];
  view?: string;
  scope_key?: string;
  version?: string;
  expanded_parent_ids?: string[];
  truncated?: boolean;
  total_actors?: number;
};

function appearance(node: any) {
  const kind = String(
    node.entity_type || node.detail?.profile?.agent_type || "",
  ).toLowerCase();
  if (["asset", "location", "place", "event"].includes(kind))
    return {
      Icon: kind === "asset" ? Package : MapPin,
      label: kind === "asset" ? "Asset or object" : "Location or event",
      accent: "#64748b",
      background: "#f1f5f9",
    };
  if (node.type === "case")
    return {
      Icon: GitBranch,
      label: "Analysis topic",
      accent: "#172137",
      background: "#f4f5fa",
    };
  if (node.type === "group")
    return {
      Icon: Users,
      label: "Affiliation unverified",
      accent: "#936017",
      background: "#fff8ec",
    };
  if (node.type === "source")
    return {
      Icon: FileText,
      label: "Source snapshot",
      accent: "#5145d7",
      background: "#f5f3ff",
    };
  if (["timeline", "simulated"].includes(node.type))
    return {
      Icon: CircleDot,
      label: node.type === "simulated" ? "Simulated event" : "Historical event",
      accent: node.type === "simulated" ? "#936017" : "#315c80",
      background: "#eef3fb",
    };
  if (/person|individual/.test(kind))
    return {
      Icon: UserRound,
      label: "Person",
      accent: "#625b8e",
      background: "#f5f0fa",
    };
  if (/group|collective/.test(kind))
    return {
      Icon: Users,
      label: "Group",
      accent: "#936017",
      background: "#fff8ec",
    };
  if (/government|country|state/.test(kind))
    return {
      Icon: Flag,
      label: "Public institution",
      accent: "#276c60",
      background: "#f0f7f4",
    };
  return {
    Icon: Building2,
    label: /company|organization|institution/.test(kind) ? "Institution" : "Participants",
    accent: "#315c80",
    background: "#f0f5fa",
  };
}

export default function LinkedKnowledgeGraph({
  graph,
  onSelect,
}: {
  graph: Graph;
  onSelect: (node: any) => void;
}) {
  const canvas = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState("");
  const [fullscreen, setFullscreen] = useState(false);
  const [zoom, setZoom] = useState(0.8),
    [pan, setPan] = useState({ x: 30, y: 30 });
  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;
  const [size, setSize] = useState({ width: 700, height: 520 });
  const gesture = useRef<{
    x: number;
    y: number;
    px: number;
    py: number;
  } | null>(null);
  const focusBranch = useRef<string | null>(null);
  const scope =
    graph.scope_key ||
    `${graph.view}:${graph.version}:${graph.nodes.find((n) => n.type === "case")?.id || graph.nodes[0]?.id || "empty"}`;
  const searchParents = JSON.stringify(graph.expanded_parent_ids || []);
  useEffect(() => {
    setExpanded(new Set(JSON.parse(searchParents)));
    focusBranch.current = JSON.parse(searchParents)[0] || null;
    setSelected("");
  }, [scope, searchParents]);
  useEffect(() => {
    const close = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);
  const children = useMemo(() => {
    const map = new Map<string, string[]>();
    if (graph.view === "actors")
      graph.edges.forEach((e) => {
        if (
          ["membership", "topic"].includes(e.kind) &&
          !String(e.source).startsWith("case:")
        )
          map.set(e.source, [...(map.get(e.source) || []), e.target]);
      });
    return map;
  }, [graph]);
  const layout = useMemo(() => {
    const hidden = new Set<string>();
    children.forEach((ids, parent) => {
      if (!expanded.has(parent)) ids.forEach((id) => hidden.add(id));
    });
    const nodes = graph.nodes.filter((n) => !hidden.has(n.id));
    const ids = new Set(nodes.map((n) => n.id));
    const edges = graph.edges.filter(
      (e) => ids.has(e.source) && ids.has(e.target),
    );
    if (!nodes.length) return { nodes: [], edges: [], width: 1, height: 1 };
    const cy = cytoscape({
      headless: true,
      styleEnabled: true,
      elements: [
        ...nodes.map((n) => ({ data: { id: n.id } })),
        ...edges.map((e, i) => ({
          data: { id: `edge_${i}`, source: e.source, target: e.target },
        })),
      ],
      style: [{ selector: "node", style: { width: NODE_WIDTH, height: NODE_HEIGHT } }],
    });
    cy.layout({
      name: "dagre",
      rankDir: "LR",
      nodeSep: 40,
      rankSep: 88,
      padding: 30,
      animate: false,
    } as any).run();
    const positions = new Map<string, { x: number; y: number }>();
    cy.nodes().forEach((n) => {
      positions.set(n.id(), {
        x: n.position("x") - NODE_WIDTH / 2,
        y: n.position("y") - NODE_HEIGHT / 2,
      });
    });
    cy.destroy();
    const positioned = nodes.map((n) => ({
      ...n,
      ...positions.get(n.id),
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
    }));
    return {
      nodes: positioned,
      edges,
      width: Math.max(...positioned.map((n) => n.x + NODE_WIDTH + 30)),
      height: Math.max(...positioned.map((n) => n.y + NODE_HEIGHT + 30)),
    };
  }, [graph, children, expanded]);
  useEffect(() => {
    if (!canvas.current) return;
    const observer = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      setSize({ width: r.width, height: r.height });
    });
    observer.observe(canvas.current);
    return () => observer.disconnect();
  }, [fullscreen]);
  const fittedScope = useRef("");
  const fittedWidth = useRef(0);
  function fit(overview = false) {
    const z = Math.min(
      1,
      Math.max(
        overview ? 0.28 : 0.68,
        Math.min(
          (size.width - 42) / layout.width,
          (size.height - 50) / layout.height,
        ),
      ),
    );
    setZoom(z);
    setSelected("");
    setPan({
      x: Math.max(22, (size.width - layout.width * z) / 2),
      y: Math.max(28, (size.height - layout.height * z) / 2),
    });
  }
  useEffect(() => {
    if (
      layout.nodes.length &&
      (fittedScope.current !== scope || fittedWidth.current !== size.width)
    ) {
      fittedScope.current = scope;
      fittedWidth.current = size.width;
      fit();
    }
  }, [scope, layout, size.width]);
  useEffect(() => {
    const id = focusBranch.current;
    if (!id) return;
    if (id === "__fit__") {
      focusBranch.current = null;
      fit();
      return;
    }
    if (!expanded.has(id)) return;
    const branch = layout.nodes.find((n) => n.id === id);
    if (!branch) return;
    const frame = requestAnimationFrame(() => {
      focusBranch.current = null;
      const currentZoom = zoomRef.current;
      setPan({
        x: 24 - branch.x * currentZoom,
        y: size.height / 2 - (branch.y + NODE_HEIGHT / 2) * currentZoom,
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [layout, scope, expanded, zoom, size.height]);
  const neighbors = new Set<string>([selected]);
  graph.edges.forEach((e) => {
    if (e.source === selected) neighbors.add(e.target);
    if (e.target === selected) neighbors.add(e.source);
  });
  function scale(factor: number) {
    const next = Math.max(0.4, Math.min(1.6, zoom * factor));
    const center = { x: size.width / 2, y: size.height / 2 };
    setPan({
      x: center.x - ((center.x - pan.x) * next) / zoom,
      y: center.y - ((center.y - pan.y) * next) / zoom,
    });
    setZoom(next);
  }
  const content = (
    <div
      className={
        fullscreen
          ? "fixed inset-3 z-[80] bg-white rounded-2xl shadow-2xl border flex flex-col p-3"
          : "knowledge-graph border overflow-hidden bg-paper-50"
      }
    >
      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-paper-200/80 text-xs text-ink-500 bg-white">
        <span className="font-medium text-ink-700 mr-auto">
          Relationship Graph{" "}
          <span className="font-normal text-ink-400 ml-2">
            {layout.nodes.length} nodes
          </span>
        </span>
        {!!children.size && (
          <button
            className="border rounded-lg px-2.5 py-1.5 hover:bg-paper-50"
            onClick={() => {
              focusBranch.current = expanded.size
                ? "__fit__"
                : Array.from(children.keys())[0] || null;
              setExpanded(
                expanded.size
                  ? new Set()
                  : new Set(Array.from(children.keys())),
              );
            }}
          >
            {expanded.size ? "Collapse groups" : "Expand groups"}
          </button>
        )}
        <button
          aria-label="Zoom out"
          className="p-1.5 rounded-lg border"
          onClick={() => scale(0.8)}
        >
          <Minus size={14} />
        </button>
        <span className="tabular-nums w-8 text-center">
          {Math.round(zoom * 100)}%
        </span>
        <button
          aria-label="Zoom in"
          className="p-1.5 rounded-lg border"
          onClick={() => scale(1.25)}
        >
          <Plus size={14} />
        </button>
        <button
          aria-label="Fit to view"
          className="p-1.5 rounded-lg border"
          onClick={() => fit(true)}
        >
          <Scan size={14} />
        </button>
        <button
          aria-label={fullscreen ? "Exit fullscreen" : "Fullscreen graph"}
          className="p-1.5 rounded-lg border"
          onClick={() => setFullscreen(!fullscreen)}
        >
          {fullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
        </button>
      </div>
      <div
        ref={canvas}
        aria-label="Interactive knowledge graph"
        className={
          fullscreen
            ? "relative flex-1 min-h-0 overflow-hidden touch-none"
            : "relative overflow-hidden h-[460px] touch-none"
        }
        style={{
          backgroundImage: "radial-gradient(#d9e2ec .6px, transparent .6px)",
          backgroundSize: "24px 24px",
        }}
        onPointerDown={(e) => {
          if ((e.target as Element).closest("button")) return;
          gesture.current = {
            x: e.clientX,
            y: e.clientY,
            px: pan.x,
            py: pan.y,
          };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const g = gesture.current;
          if (g)
            setPan({ x: g.px + e.clientX - g.x, y: g.py + e.clientY - g.y });
        }}
        onPointerUp={() => {
          gesture.current = null;
        }}
        onPointerCancel={() => {
          gesture.current = null;
        }}
      >
        <div
          style={{
            position: "absolute",
            transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`,
            transformOrigin: "0 0",
            width: layout.width,
            height: layout.height,
          }}
        >
          <svg
            width={layout.width}
            height={layout.height}
            className="absolute inset-0 pointer-events-none"
            aria-hidden="true"
          >
            <defs>
              <marker
                id="knowledge-arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#9baec1" />
              </marker>
            </defs>
            {layout.edges.map((e, i) => {
              const a = layout.nodes.find((n) => n.id === e.source),
                b = layout.nodes.find((n) => n.id === e.target);
              if (!a || !b) return null;
              const x = a.x + a.width,
                y = a.y + NODE_HEIGHT / 2,
                tx = b.x,
                ty = b.y + NODE_HEIGHT / 2,
                m = (x + tx) / 2;
              return (
                <g
                  key={i}
                  opacity={
                    selected &&
                    !neighbors.has(e.source) &&
                    !neighbors.has(e.target)
                      ? 0.2
                      : 1
                  }
                >
                  <title>
                    {e.kind === "membership"
                      ? "Inferred affiliation · unverified"
                      : e.label === "participant"
                        ? "Event participant"
                        : e.label === "exact_span"
                          ? "Source evidence"
                          : e.label === "page_only"
                            ? "Page-level source"
                            : e.label || "Linked"}
                  </title>
                  <path
                    d={`M ${x} ${y} C ${m} ${y}, ${m} ${ty}, ${tx - 4} ${ty}`}
                    style={{ pointerEvents: "stroke" }}
                    fill="none"
                    stroke={
                      e.status === "model_inferred" ? "#aaa3bf" : "#a8b8c8"
                    }
                    strokeWidth="1.4"
                    strokeDasharray={
                      e.status === "model_inferred" ? "5 5" : undefined
                    }
                    markerEnd="url(#knowledge-arrow)"
                  />
                </g>
              );
            })}
          </svg>
          {layout.nodes.map((node) => {
            const a = appearance(node),
              Icon = a.Icon,
              p = node.detail?.profile || {},
              isBranch = children.has(node.id);
            let name = String(node.label || node.id);
            if (node.type === "source") {
              try {
                name = new URL(node.detail?.url || name).hostname;
              } catch {
                name = "Source snapshot";
              }
            }
            const detail =
              node.type === "timeline" || node.type === "simulated"
                ? node.detail?.timestamp
                : node.type === "case"
                  ? "Events · Entities · Sources"
                  : isBranch
                    ? `${children.get(node.id)!.length} members or related entities`
                    : [p.title, p.affiliation].find(
                        (value) =>
                          typeof value === "string" &&
                          value.trim() &&
                          value.trim().toLowerCase() !==
                            name.trim().toLowerCase(),
                      ) ||
                      p.activity_summary ||
                      (
                        {
                          principal: "Independent decision actor",
                          member: "Member · affiliation unverified",
                          mentioned: "Participant in source material",
                        } as Record<string, string>
                      )[node.decision_role] ||
                      "Open profile and evidence";
            return (
              <button
                key={node.id}
                aria-label={`View ${name}`}
                onClick={() => {
                  setSelected(node.id);
                  if (isBranch) {
                    focusBranch.current = expanded.has(node.id)
                      ? "__fit__"
                      : node.id;
                    setExpanded((old) => {
                      const next = new Set(old);
                      next.has(node.id)
                        ? next.delete(node.id)
                        : next.add(node.id);
                      return next;
                    });
                  }
                  if (node.type === "actor") setFullscreen(false);
                  onSelect(node);
                }}
                className="knowledge-graph-node absolute text-left bg-white transition-shadow overflow-hidden"
                style={{
                  left: node.x,
                  top: node.y,
                  width: NODE_WIDTH,
                  height: NODE_HEIGHT,
                  border: `${selected === node.id ? 2 : 1}px solid ${selected === node.id ? a.accent : "#dbe3ec"}`,
                  borderLeft: `4px solid ${a.accent}`,
                  opacity: selected && !neighbors.has(node.id) ? 0.35 : 1,
                }}
              >
                <div className="flex gap-2.5 px-3.5 pt-3.5">
                  <span
                    className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                    style={{ background: a.background, color: a.accent }}
                  >
                    <Icon size={17} />
                  </span>
                  <div className="min-w-0">
                    <span
                      className="text-xs font-medium"
                      style={{ color: a.accent }}
                    >
                      {a.label}
                      {node.decision_role === "principal" ? " · Decision actor" : ""}
                    </span>
                    <p className="text-[14px] text-ink-800 font-semibold leading-snug line-clamp-2">
                      {name}
                    </p>
                  </div>
                  {isBranch && (
                    <span className="ml-auto text-ink-400">
                      {expanded.has(node.id) ? "−" : "+"}
                    </span>
                  )}
                </div>
                <p className="px-3.5 mt-2 text-xs text-ink-500 truncate">
                  {detail}
                </p>
              </button>
            );
          })}
        </div>
        {!layout.nodes.length && (
          <p className="absolute inset-0 flex items-center justify-center text-sm text-ink-400">
            No matching relationships
          </p>
        )}
        <span className="absolute bottom-3 left-3 text-xs text-ink-400 bg-white/90 rounded px-2 py-1 pointer-events-none">
          Drag to pan · Select a card for details
        </span>
      </div>
      <div className="px-4 py-2.5 bg-white border-t flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-500">
        <span>
          <i className="inline-block w-2 h-2 rounded-full bg-brand-500 mr-1.5" />
          Institution
        </span>
        <span>
          <i className="inline-block w-2 h-2 rounded-full bg-violet-400 mr-1.5" />
          Person
        </span>
        <span>
          <i className="inline-block w-2 h-2 rounded-full bg-amber-500 mr-1.5" />
          Group or unassigned
        </span>
        <span>
          <i className="inline-block w-2 h-2 rounded-full bg-emerald-600 mr-1.5" />
          Source
        </span>
        <span>
          <i className="inline-block w-2 h-2 rounded-full bg-paper-400 mr-1.5" />
          Asset or location
        </span>
        <span className="ml-auto">Dashed lines indicate inferred affiliations. Arrows show link direction.</span>
      </div>
      {graph.truncated && (
        <p className="px-4 py-2 text-xs text-amber-700 bg-amber-50">
          The display limit is reached. Search to find more entities.
        </p>
      )}
    </div>
  );
  return fullscreen && typeof document !== "undefined"
    ? createPortal(content, document.body)
    : content;
}
