const $ = (id) => document.getElementById(id);
const langs = Object.fromEntries(window.LANGUAGES.map((l) => [l.key, l]));
let direction = "en2x";

const EXAMPLES = [
  "How are you?",
  "The farmers of this village grow rice and vegetables.",
  "Education is the key to a better future.",
];

function current() {
  return langs[$("lang").value];
}

function refreshLabels() {
  const l = current();
  $("srcLabel").textContent = direction === "en2x" ? "English" : l.name;
  $("tgtLabel").textContent = direction === "en2x" ? l.name : "English";

  const badges = [];
  badges.push(`<span class="badge">Category ${l.category}</span>`);
  badges.push(l.native
    ? `<span class="badge ok">In NLLB-200 pre-training</span>`
    : `<span class="badge warn">Not in NLLB-200: needs fine-tuning</span>`);
  badges.push(l.finetuned
    ? `<span class="badge ok">Fine-tuned model ready</span>`
    : `<span class="badge">No fine-tuned model yet</span>`);
  $("langInfo").innerHTML = badges.join("");

  const examples = direction === "en2x" ? EXAMPLES
    : (l.results.finetuned || l.results.baseline || {}).samples?.slice(0, 3).map((s) => s.other) || [];
  $("examples").innerHTML = examples
    .map((e, i) => `<button class="chip" data-i="${i}" title="${escapeHtml(e)}">${escapeHtml(e.length > 40 ? e.slice(0, 40) + "…" : e)}</button>`)
    .join("");
  $("examples").querySelectorAll(".chip").forEach((b) =>
    b.addEventListener("click", () => { $("source").value = examples[b.dataset.i]; updateCount(); translate(); }));
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function updateCount() {
  $("count").textContent = `${$("source").value.length} / ${$("source").maxLength}`;
}

function setStatus(msg, kind = "") {
  $("status").textContent = msg;
  $("status").className = `status ${kind || "muted"}`;
}

async function translate() {
  const text = $("source").value.trim();
  if (!text) { setStatus("Enter some text first."); return; }
  const btn = $("translate");
  btn.disabled = true;
  setStatus("Translating… (the first request loads the model and can take a minute)");
  const slow = setTimeout(() => setStatus("Still working… loading/downloading the model (~2.4 GB the first time)."), 8000);
  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang: $("lang").value, direction, model: $("model").value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    $("target").value = data.translation;
    setStatus(`${data.direction} · ${data.model} · ${data.seconds}s`, "ok");
  } catch (err) {
    $("target").value = "";
    setStatus(err.message, "error");
  } finally {
    clearTimeout(slow);
    btn.disabled = false;
  }
}

$("lang").addEventListener("change", () => { $("target").value = ""; setStatus(""); refreshLabels(); });
$("swap").addEventListener("click", () => {
  direction = direction === "en2x" ? "x2en" : "en2x";
  const out = $("target").value;
  if (out) { $("source").value = out; $("target").value = ""; }
  updateCount();
  refreshLabels();
});
$("translate").addEventListener("click", translate);
$("source").addEventListener("input", updateCount);
$("source").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) translate(); });
$("copy").addEventListener("click", async () => {
  if (!$("target").value) return;
  try { await navigator.clipboard.writeText($("target").value); setStatus("Copied.", "ok"); } catch { /* ignore */ }
});

refreshLabels();
