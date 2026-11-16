const utcDate = new Intl.DateTimeFormat("en-US", { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" });
const nonWord = new RegExp("[^\\p{L}\\p{N}]+", "gu");

export function formatReportDate(value: string) {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? `${utcDate.format(timestamp)} UTC` : value;
}

export function normalizeReportText(value: string) {
  return value.normalize("NFKC").toLowerCase().replace(nonWord, " ").trim();
}

export function similarReportText(a: string, b: string) {
  const left = normalizeReportText(a);
  const right = normalizeReportText(b);
  return left === right || (Math.min(left.length, right.length) >= 40 && (left.includes(right) || right.includes(left)));
}

export const verdictLabels = { yes: "Yes criterion met", no: "No criterion met", unresolved: "Unresolved", estimated: "Conditional estimate" };
