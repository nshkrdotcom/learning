function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "-"
  const number = Number(value)
  if (Math.abs(number) >= 1000 || (Math.abs(number) < 0.001 && number !== 0)) {
    return number.toExponential(2)
  }
  return number.toFixed(4).replace(/\.?0+$/, "")
}

function titleCase(value) {
  return String(value || "").replace(/_/g, " ").replace(/\b\w/g, letter => letter.toUpperCase())
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
}

export {clamp, escapeHtml, formatNumber, titleCase}

