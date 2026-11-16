export const BACKEND_STORAGE_KEY = "eip_backend_api_url";

export function normalizeBackendUrl(value: string): string {
  const input = value.trim();
  if (!input) return "";
  const url = new URL(input);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error("Enter an HTTP or HTTPS backend URL without credentials, a query string, or a fragment.");
  }
  if (typeof window !== "undefined" && window.location.protocol === "https:" && url.protocol !== "https:") {
    throw new Error("Hosted sites require an HTTPS backend. Use the local site to connect to a local HTTP backend.");
  }
  const path = url.pathname.replace(/\/+$/, "");
  url.pathname = path.endsWith("/api") ? path : `${path}/api`;
  return url.toString().replace(/\/+$/, "");
}

export function readBackendUrl(): string {
  if (typeof window === "undefined") return "";
  const configured = localStorage.getItem(BACKEND_STORAGE_KEY) || process.env.NEXT_PUBLIC_API_URL;
  if (configured) {
    try { return normalizeBackendUrl(configured); } catch { return ""; }
  }
  return ["localhost", "127.0.0.1"].includes(window.location.hostname)
    ? "http://localhost:8000/api" : "";
}
