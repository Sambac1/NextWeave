"""Agent-assisted profile retrieval with cited local evidence and optional web tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

from pydantic import Field

from app.layers.retrieval.sdk_runtime import SDKStrictModel, _run_structured_agent
from app.layers.support.semantic_view import normalize_semantic_view

_TITLE_ALIASES = {
    "中美贸易战": "China–United States trade war",
    "china us trade war": "China–United States trade war",
    "china-united states trade war": "China–United States trade war",
}

WEB_RETRIEVAL_ENABLED = False


class SDKProfileSummary(SDKStrictModel):
    name: str = ""
    role: str = ""
    stance: str = ""
    entity_type: str = ""
    title: str = ""
    affiliation: str = ""
    importance_score: float = 0.0
    summary: str = ""
    key_actions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class SDKRetrievalAnswer(SDKStrictModel):
    answer: str = ""
    profiles: list[SDKProfileSummary] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class SDKMaterialAnswer(SDKStrictModel):
    """Per-principal material retrieval for the profile-generation stage.

    The SDK agent pulls every local passage about ONE figure (timeline events,
    background chapters, its own profile card) and returns them as verbatim,
    source-tagged excerpts — the raw evidence the profile LLM synthesizes from
    (task 2, classmate's Agent-SDK retrieval wired into the pipeline).
    """
    material: str = ""
    sources: list[str] = Field(default_factory=list)


def _artifact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def resolve_event_dir(title: str) -> Path | None:
    """Resolve an event title to its local artifact directory."""
    resolved = _TITLE_ALIASES.get(title.strip().lower(), title)

    from app.layers.support.llm_artifacts import _event_artifact_dir

    found = _event_artifact_dir(resolved)
    if found is not None and found.is_dir():
        return found

    return None


def _clip(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _block(source_id: str, body: str) -> str:
    """Attach a precise source marker to a retrieved evidence block."""
    return f"[source: {source_id}]\n{body}"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _load_profiles(event_dir: Path) -> list[dict]:
    """Load the profile set the retrieval agent searches over."""
    path = event_dir / "profiles.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _bdi_block_name(profile: dict) -> str:
    """Display name of a profiles.json entry (principal or member)."""
    return (
        profile.get("source_figure")
        or profile.get("canonical_name")
        or profile.get("name")
        or profile.get("agent_id")
        or "?"
    )


def bdi_profile_to_text(profile: dict, max_history: int = 12) -> str:
    """Format a profiles.json entry"""
    if not isinstance(profile, dict):
        return ""
    if str(profile.get("agent_level") or "principal").lower() == "member":
        return _member_to_text(profile)
    return _principal_to_text(profile, max_history)


def _member_to_text(member: dict) -> str:
    lines = [f"Name: {_bdi_block_name(member)}"]
    if member.get("entity_type"):
        lines.append(f"Entity: {_clip(member.get('entity_type'), 80)}")
    if member.get("role_in_entity"):
        lines.append(f"Role in parent: {_clip(member.get('role_in_entity'), 160)}")
    if member.get("title"):
        lines.append(f"Title: {_clip(member.get('title'), 140)}")
    if member.get("affiliation"):
        lines.append(f"Affiliation: {_clip(member.get('affiliation'), 140)}")
    if member.get("stance"):
        lines.append(f"Stance: {_clip(member.get('stance'), 140)}")
    parent = member.get("parent_agent_id")
    if parent:
        lines.append(f"Member of: {parent}")
    return "\n".join(lines)


def _principal_to_text(profile: dict, max_history: int) -> str:
    """Format a principal (BDI actor) profile into the compact retrieval block."""
    semantic = profile.get("semantic") if isinstance(profile.get("semantic"), dict) else {}
    view = normalize_semantic_view(semantic)
    lines = [f"Name: {_bdi_block_name(profile)}"]
    if profile.get("entity_type"):
        lines.append(f"Entity: {_clip(profile.get('entity_type'), 80)}")
    if profile.get("agent_level"):
        lines.append(f"Level: {_clip(profile.get('agent_level'), 20)}")
    if profile.get("knowledge_boundary"):
        lines.append(f"Knowledge boundary: {_clip(profile.get('knowledge_boundary'), 40)}")

    # Identity — the individual form fills position / party, the institution
    # and collective forms fill nature / event_role.
    identity = view.get("identity") or {}
    parts = []
    for k in ("nature", "event_role", "position", "party"):
        val = identity.get(k)
        if isinstance(val, str) and val:
            parts.append(val)
    if parts:
        lines.append(f"Identity: {'; '.join(_clip(p, 200) for p in parts)}")
    # Composition — member-based organisations list their constituents here.
    constituents = identity.get("composition") or identity.get("constituents")
    if isinstance(constituents, list) and constituents:
        lines.append(f"Composition: {_clip(', '.join(str(c) for c in constituents), 300)}")

    # Views / stance.
    views = view.get("views") or {}
    if views.get("stance"):
        lines.append(f"Stance: {_clip(views['stance'], 400)}")

    # Core concerns — priorities the actor cares about most.
    concerns = view.get("concerns") or {}
    prio = concerns.get("priority_order")
    if isinstance(prio, list) and prio:
        lines.append(f"Core concerns: {'; '.join(_clip(str(p), 140) for p in prio[:5])}")

    # Experience — career for individuals, evolution for institutions.
    experience = view.get("experience") or {}
    career = experience.get("career_summary") or experience.get("evolution")
    if career:
        lines.append(f"Experience: {_clip(career, 300)}")

    # grand_goals
    goals = view.get("goals")
    if isinstance(goals, list) and goals:
        goal_lines = []
        for g in goals[:3]:
            if isinstance(g, dict) and g.get("goal"):
                goal_lines.append(_clip(str(g.get("goal")), 160))
        if goal_lines:
            lines.append("Grand goals: " + "; ".join(goal_lines))

    # Strength — PERSON power_base / ENTITY capabilities.
    strength = view.get("strength") or {}
    cap_lines = []
    for k, v in strength.items():
        if isinstance(v, dict):
            lvl = v.get("level") or v.get("status") or ""
            desc = v.get("description") or ""
            if lvl or desc:
                cap_lines.append(f"{k}: {lvl}{(' - ' + _clip(str(desc), 160)) if desc else ''}")
        else:
            cap_lines.append(f"{k}: {_clip(str(v), 160)}")
    if cap_lines:
        lines.append("Capabilities: " + " | ".join(cap_lines[:6]))

    # Style — the preferred-tools sub-block, read through the shared style view.
    preferences = view.get("preferences") or {}
    if preferences.get("action_pace"):
        lines.append(f"Action pace: {_clip(preferences['action_pace'], 60)}")
    tools = preferences.get("preferred_tools")
    if isinstance(tools, list) and tools:
        lines.append(f"Preferred tools: {', '.join(_clip(str(t), 40) for t in tools[:8])}")

    # episodic history
    episodic = profile.get("episodic")
    if isinstance(episodic, dict):
        history = episodic.get("history")
        if isinstance(history, list) and history:
            lines.append("History:")
            for h in history[:max_history]:
                if isinstance(h, dict):
                    ts = h.get("timestamp") or ""
                    desc = h.get("description") or ""
                    if desc:
                        lines.append(f"  - [{ts}] {_clip(desc, 220)}")

    # subordinates (member tree, names only)
    subs = profile.get("subordinates")
    if isinstance(subs, list) and subs:
        names = [_bdi_block_name(s) for s in subs if isinstance(s, dict)]
        if names:
            lines.append(f"Subordinates: {', '.join(names[:10])}")
    return "\n".join(lines)


def _load_timeline(event_dir: Path) -> list[dict]:
    from app.layers.support.llm_artifacts import _load_timeline_from_dir

    return _load_timeline_from_dir(event_dir)


def _load_background_summaries(event_dir: Path) -> list[dict]:
    """Load section summaries with their source paths for background retrieval."""
    from app.layers.support.llm_artifacts import _load_background_summaries_from_dir

    items = _load_background_summaries_from_dir(event_dir)
    return [
        {"source": item["source"], "text": _clip(item["text"], 700)}
        for item in items
        if item.get("text")
    ]


def timeline_to_text(ev: dict) -> str:
    from app.layers.support.llm_artifacts import timeline_actor_names

    ts = ev.get("timestamp") or ev.get("date_original") or ""
    desc = ev.get("description") or ""
    actors = timeline_actor_names(ev)
    lines = [f"[{ts}] {_clip(desc, 400)}"]
    if actors:
        lines.append(f"  Actors: {', '.join(str(a) for a in actors)}")
    return "\n".join(lines)


def _score_by_query(text: str, query: str, base: float = 0.0) -> float:
    terms = set(re.findall(r"\w+", (query or "").lower()))
    if not terms:
        return base
    lowered = text.lower()
    return sum(1 for t in terms if t in lowered) + base


def _rank_profiles(profiles: list[dict], query: str, top_k: int) -> list[dict]:
    if not query.strip():
        return profiles[:top_k]
    scored = [
        (_score_by_query(bdi_profile_to_text(p, max_history=3), query), p)
        for p in profiles
    ]
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_k]]


def _rank_timeline(events: list[dict], query: str, top_k: int) -> list[dict]:
    from app.layers.support.llm_artifacts import timeline_actor_names, timeline_actor_strength

    scored = []
    for ev in events:
        text = (ev.get("description", "") + " " + " ".join(timeline_actor_names(ev)))
        base = float(ev.get("importance_score") or 0.0) * 0.5
        scored.append((_score_by_query(text, query, base) + 0.5 * timeline_actor_strength(ev), ev))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_k]]


def _rank_background(items: list[dict], query: str, top_k: int) -> list[dict]:
    if not query.strip():
        return items[:top_k]
    scored = [(_score_by_query(it["text"], query), it) for it in items]
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:top_k]]


def _ddg_decode(href: str) -> str:
    if "uddg=" in href:
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            return urllib.parse.unquote(m.group(1))
    return href


async def _search_tavily(query: str, top_k: int, api_key: str) -> list[dict]:
    import httpx

    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": top_k},
        )
        resp.raise_for_status()
        data = resp.json()
    out: list[dict] = []
    for it in data.get("results", [])[:top_k]:
        out.append({
            "title": it.get("title", ""),
            "url": it.get("url", ""),
            "snippet": it.get("content") or it.get("snippet", ""),
        })
    return out


async def _search_ddg(query: str, top_k: int) -> list[dict]:
    import httpx

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }
    try:
        async with httpx.AsyncClient(
            timeout=25.0, headers=headers, proxy=proxy, follow_redirects=True
        ) as client:
            resp = await client.post("https://lite.duckduckgo.com/lite/", data={"q": query})
            html = resp.text
    except Exception:
        return []

    titles = re.findall(
        r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S
    )
    snippets = re.findall(
        r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', html, re.S
    )
    out: list[dict] = []
    for i in range(min(top_k, len(titles))):
        href, title = titles[i]
        snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
        out.append({
            "title": _strip_html(title),
            "url": _ddg_decode(href),
            "snippet": snippet,
        })
    return out


async def _search_wikipedia(query: str, top_k: int) -> list[dict]:
    """Search Wikipedia through the MediaWiki API."""
    import httpx

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    headers = {
        "User-Agent": (
            "EverythingIsPossible/1.0 (political prediction sandbox; "
            "https://github.com/example/eip)"
        ),
    }
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": str(top_k),
        "srprop": "snippet",
    }
    try:
        async with httpx.AsyncClient(
            timeout=25.0, headers=headers, proxy=proxy, follow_redirects=True
        ) as client:
            resp = await client.get("https://en.wikipedia.org/w/api.php", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return []

    out: list[dict] = []
    for it in (data.get("query", {}) or {}).get("search", []):
        title = it.get("title", "")
        if not title:
            continue
        out.append({
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "snippet": _strip_html(it.get("snippet", "")),
        })
    return out


def _bing_decode(url: str) -> str:
    """Decode a Bing redirect into its original destination URL."""
    m = re.search(r"[?&]u=a1([^&]+)", url)
    if m:
        try:
            import base64

            return base64.urlsafe_b64decode(m.group(1) + "==").decode("utf-8", "ignore")
        except Exception:
            pass
    return url


def _parse_bing(html: str, top_k: int) -> list[dict]:
    results: list[dict] = []
    for part in re.split(r'<li class="b_algo"', html)[1:]:
        m = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', part, re.S)
        if not m:
            continue
        title = _strip_html(m.group(2))
        if not title:
            continue
        snippet = ""
        p = re.search(r'<p[^>]*>(.*?)</p>', part, re.S)
        if p:
            snippet = _strip_html(p.group(1))
        results.append({
            "title": title,
            "url": _bing_decode(m.group(1)),
            "snippet": snippet,
        })
        if len(results) >= top_k:
            break
    return results


async def _search_bing(query: str, top_k: int) -> list[dict]:
    """Search Bing without requiring an API credential."""
    import httpx

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }
    try:
        async with httpx.AsyncClient(
            timeout=25.0, headers=headers, proxy=proxy, follow_redirects=True
        ) as client:
            resp = await client.get(
                "https://cn.bing.com/search", params={"q": query, "count": str(top_k)}
            )
            html = resp.text
    except Exception:
        return []
    return _parse_bing(html, top_k)


async def _search_web(query: str, top_k: int = 5) -> list[dict]:
    """Search configured providers with Bing, Wikipedia, and DuckDuckGo fallbacks."""
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            return await _search_tavily(query, top_k, tavily_key)
        except Exception:
            pass
    results = await _search_bing(query, top_k)
    if results:
        return results
    results = await _search_wikipedia(query, top_k)
    if results:
        return results
    return await _search_ddg(query, top_k)


SEARCH_LOCAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "检索关键词或问题，用于匹配画像/时间线/背景。",
        },
        "top_k": {
            "type": "integer",
            "description": "返回条数，默认 8。",
        },
        "kind": {
            "type": "string",
            "description": "检索源类型：profiles(人物画像) / timeline(时间线) / background(章节摘要) / all(全部)，默认 profiles。",
        },
    },
    "required": ["query"],
}

SEARCH_WEB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索关键词。",
        },
        "top_k": {
            "type": "integer",
            "description": "返回结果数，默认 5。",
        },
    },
    "required": ["query"],
}


def _make_search_local_tool(event_dir: Path):
    """Build the cited local-evidence retrieval tool."""

    async def handler(args: Any) -> dict:
        query = str((args or {}).get("query") or "").strip()
        top_k = int((args or {}).get("top_k") or 8)
        kind = str((args or {}).get("kind") or "profiles").strip().lower()

        blocks: list[str] = []
        if kind in ("profiles", "all"):
            profiles = _load_profiles(event_dir)
            for p in _rank_profiles(profiles, query, top_k):
                name = p.get("canonical_name") or p.get("figure") or p.get("name") or p.get("source_figure") or "?"
                # Every entry here comes from profiles.json, so the block is
                # formatted by shape (principal semantic / member card) and
                # tagged with one source.
                text = bdi_profile_to_text(p)
                blocks.append(_block(f"profiles.json/{name}", text))
        if kind in ("timeline", "all"):
            for ev in _rank_timeline(_load_timeline(event_dir), query, top_k):
                tid = ev.get("timeline_id") or ev.get("timestamp") or "?"
                blocks.append(_block(f"timeline.json#{tid}", timeline_to_text(ev)))
        if kind in ("background", "all"):
            for item in _rank_background(_load_background_summaries(event_dir), query, top_k):
                blocks.append(_block(item["source"], item["text"]))

        if not blocks:
            return {
                "content": [{
                    "type": "text",
                    "text": "NO_RESULTS: 本地未检索到相关内容，可尝试更换 kind 或 query。",
                }],
            }
        return {"content": [{"type": "text", "text": "\n\n".join(blocks)}]}

    return handler


def _make_search_web_tool():
    """Build the web-search tool; return an unavailable result when web retrieval is disabled."""

    async def handler(args: Any) -> dict:
        if not WEB_RETRIEVAL_ENABLED:
            return {
                "content": [{
                    "type": "text",
                    "text": "WEB_SEARCH_UNAVAILABLE: 网络检索当前不可用，请使用 search_local 检索本地材料。",
                }],
            }
        query = str((args or {}).get("query") or "").strip()
        top_k = int((args or {}).get("top_k") or 5)
        if not query:
            return {"content": [{"type": "text", "text": "NO_QUERY"}]}
        results = await _search_web(query, top_k)
        if not results:
            return {
                "content": [{
                    "type": "text",
                    "text": "WEB_SEARCH_UNAVAILABLE: 网络检索失败或无结果（可能被墙或需代理），请改用 search_local。",
                }],
            }
        blocks = [
            _block(r.get("url", "?"), f"Title: {r.get('title', '')}\nSnippet: {_clip(r.get('snippet', ''), 400)}")
            for r in results
        ]
        return {"content": [{"type": "text", "text": "\n\n".join(blocks)}]}

    return handler


async def run_retrieval_agent(
    *,
    api_key: str,
    model: str,
    title: str,
    task: str,
    base_url: str | None = None,
) -> dict:
    """Retrieve cited profiles and restrict returned actors to the local evidence whitelist."""
    event_dir = resolve_event_dir(title)
    if event_dir is None:
        raise FileNotFoundError(f"The data folder for {title} could not be found.")

    search_local = _make_search_local_tool(event_dir)
    search_web = _make_search_web_tool()
    # Web tool exposure is gated by WEB_RETRIEVAL_ENABLED: while off (default) the tool
    # is not registered at all — the agent never wastes a turn on an unavailable tool.
    tools = [
        (
            "search_local",
            "Retrieve local event material (profiles / timeline / background) by keyword query.",
            SEARCH_LOCAL_SCHEMA,
            search_local,
        ),
    ]
    if WEB_RETRIEVAL_ENABLED:
        tools.append(
            (
                "search_web",
                "Search the web for external information about the event or figures.",
                SEARCH_WEB_SCHEMA,
                search_web,
            ),
        )

    instructions = (
        "You are a profile-retrieval and summarization agent. Use the search_local tool "
        + ("(and search_web for external info) " if WEB_RETRIEVAL_ENABLED else "")
        + "to gather material, then synthesize a "
        "structured summary. Base everything only on retrieved content — do not invent "
        "names, roles, stances, actions, or dates that do not appear in the retrieved text. "
        "Every retrieved block starts with a [source: ...] marker; use those markers to "
        "populate the 'sources' fields so every claim is traceable."
    )
    prompt = f"""
Event title: {title}
Task: {task}

Steps:
1. Call search_local with kind="profiles" to retrieve the key figure profiles.
2. If the task needs more depth or broader coverage, call search_local again with
   kind="timeline" or kind="background" (or kind="all").
3. Produce the final structured result and submit it via the output tool.

Requirements for the final result:
- "answer": an overall synthesis of the key figures' profiles for this event.
- "profiles": one entry per retrieved key figure, with name / role / stance /
  entity_type / title / affiliation / importance_score, a short "summary",
  "key_actions", and "sources" (the [source: ...] markers backing that figure).
- "sources": top-level list of every source marker you drew on.
Do not fabricate. If a field is absent from the retrieved material, leave it empty
rather than guessing.
""".strip()

    output = await _run_structured_agent(
        api_key=api_key,
        model=model,
        base_url=base_url,
        name="ProfileRetrievalAgent",
        instructions=instructions,
        output_type=SDKRetrievalAnswer,
        prompt=prompt,
        extra_tools=tools,
        max_turns=16,
    )
    # Hallucination guard: only names the local store actually holds survive.
    return _retrieval_whitelisted(output.model_dump(), _known_profile_names(event_dir))


def _name_key(value: object) -> str:
    """Normalized comparison key for names (case / internal whitespace)."""
    return " ".join(str(value or "").split()).lower()


def _known_profile_names(event_dir: Path) -> set[str]:
    """Every name the local profile store legitimately holds (principals +"""
    names: set[str] = set()
    for p in _load_profiles(event_dir):
        for key in ("source_figure", "canonical_name", "figure", "name"):
            val = p.get(key)
            if val:
                names.add(_name_key(val))
        if isinstance(p.get("aliases"), list):
            names.update(_name_key(a) for a in p["aliases"] if a)
    return names


def _retrieval_whitelisted(result: dict, known: set[str]) -> dict:
    """Drop any retrieved profile whose name is not in the local store."""
    answer = str(result.get("answer") or "")
    profiles = []
    dropped = []
    for p in result.get("profiles") or []:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        if name and _name_key(name) in known:
            profiles.append(p)
        elif name:
            dropped.append(name)
    if dropped:
        answer = (answer.rstrip() + "\n\n[note] 本地画像未收录以下提及人物，未列入："
                  + "、".join(sorted(set(dropped))) + "。").strip()
    return {"answer": answer, "profiles": profiles, "sources": result.get("sources") or []}


# Per-principal SDK retrieval budget. A hung SDK run must never hold the
# (serial) profile pipeline — on timeout the caller falls back to keyword
# retrieval, same path as any other failure.
PRINCIPAL_RETRIEVAL_TIMEOUT_SECONDS = 240


async def retrieve_principal_material(
    *,
    api_key: str,
    model: str,
    event_title: str,
    figure_name: str,
    base_url: str | None = None,
) -> dict:
    """One SDK retrieval pass scoped to ONE principal: pull every local passage
    about it and return source-tagged material for the profile LLM (task 2,
    classmate's Agent-SDK retrieval wired into the pipeline; web disabled).

    Local-only (search_local): timeline events + background chapters + its own
    profile card. Never raises — returns {"material": "", "sources": []} on any
    failure or timeout (see PRINCIPAL_RETRIEVAL_TIMEOUT_SECONDS) so the caller
    can fall back to keyword retrieval.
    """
    try:
        event_dir = resolve_event_dir(event_title)
        if event_dir is None:
            return {"material": "", "sources": []}
        search_local = _make_search_local_tool(event_dir)

        instructions = (
            "You are a retrieval agent gathering source material about ONE political "
            "figure for a profile writer. Use search_local (kind=timeline / background / "
            "profiles / all) to find every local passage mentioning the figure. "
            "Return the passages VERBATIM as retrieved — do not paraphrase, summarize, "
            "or invent content. Only content present in the retrieved blocks may be "
            "returned. Preserve the [source: ...] markers so each passage is traceable."
        )
        prompt = f"""Event: {event_title}
Figure to gather material about: {figure_name}

Steps:
1. Call search_local with kind="timeline" and a query naming the figure to collect their timeline events.
2. Call search_local with kind="background" to collect chapter summaries mentioning them.
3. Call search_local with kind="profiles" only if their own profile card is needed for context.
4. Combine every retrieved passage into the "material" field, verbatim, each passage on its own line with its [source: ...] marker.
Return only retrieved content — no commentary, no external knowledge."""
        output = await asyncio.wait_for(
            _run_structured_agent(
                api_key=api_key,
                model=model,
                base_url=base_url,
                name="MaterialRetrievalAgent",
                instructions=instructions,
                output_type=SDKMaterialAnswer,
                prompt=prompt,
                extra_tools=[
                    (
                        "search_local",
                        "Retrieve local event material (profiles / timeline / background) by keyword query.",
                        SEARCH_LOCAL_SCHEMA,
                        search_local,
                    ),
                ],
                max_turns=10,
            ),
            timeout=PRINCIPAL_RETRIEVAL_TIMEOUT_SECONDS,
        )
        return {"material": output.material or "", "sources": output.sources or []}
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "SDK material retrieval failed for %s, falling back to keyword retrieval: %s",
            figure_name, exc,
        )
        return {"material": "", "sources": []}
