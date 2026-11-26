let expPayload = null;
let activeKind = "all";
let selectedRunIds = [];
let liveApiAvailable = false;

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));
}

document.addEventListener("DOMContentLoaded", () => {
  bindControls();
  loadExperiments();
});

function bindControls() {
  document.getElementById("expSearchInput")?.addEventListener("input", renderTable);
  document.getElementById("runExperimentBtn")?.addEventListener("click", runLiveExperiment);
  document.getElementById("clearCompareBtn")?.addEventListener("click", () => {
    selectedRunIds = [];
    renderCompare();
    renderTable();
  });

  const tabs = document.querySelectorAll("#kindTablist .exp-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      activeKind = tab.dataset.kind || "all";
      tabs.forEach(t => t.setAttribute("aria-selected", String(t === tab)));
      renderTable();
    });
  });
}

async function loadExperiments() {
  try {
    const apiRes = await fetch("/api/experiments", { cache: "no-store" });
    if (apiRes.ok) {
      expPayload = await apiRes.json();
      liveApiAvailable = true;
      document.getElementById("modeBadge").innerText = "Live API Connected";
      renderAll();
      return;
    }
  } catch (_) {}

  try {
    const staticRes = await fetch("./experiments.json", { cache: "no-store" });
    if (!staticRes.ok) throw new Error(`HTTP ${staticRes.status}`);
    expPayload = await staticRes.json();
    liveApiAvailable = false;
    document.getElementById("modeBadge").innerText = "GitHub Pages Snapshot";
    renderAll();
  } catch (err) {
    document.getElementById("modeBadge").innerText = "Load Error";
    document.getElementById("expTableBody").innerHTML = `<tr><td colspan="11">Failed to load experiments data: ${escapeHtml(err.message)}</td></tr>`;
  }
}

async function runLiveExperiment() {
  if (!liveApiAvailable) return;
  const btn = document.getElementById("runExperimentBtn");
  const statusEl = document.getElementById("expStatusMsg");
  const cfg = document.getElementById("expConfigSelect").value;
  const limit = parseInt(document.getElementById("expLimitSelect").value, 10) || 100;
  btn.disabled = true;
  statusEl.innerText = `Running held-out benchmark for ${cfg} (${limit} records)...`;
  try {
    const res = await fetch("/api/experiments/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_name: cfg, sample_limit: limit }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    expPayload = data.dashboard;
    renderAll();
    statusEl.innerText = `Logged ${data.run.run_id}: ${data.run.exact_match_pct}% Exact Match across ${limit} records.`;
  } catch (err) {
    statusEl.innerText = `Experiment failed: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
}

function renderAll() {
  if (!expPayload) return;
  const summary = expPayload.summary || {};
  document.getElementById("generatedAtBadge").innerText = `Updated: ${expPayload.generated_at || "N/A"}`;
  document.getElementById("kpiTotalRuns").innerText = summary.total_runs ?? 0;
  document.getElementById("kpiBestEm").innerText = `${summary.best_exact_match_pct ?? 0}%`;
  document.getElementById("kpiTrainLoss").innerText = summary.lowest_train_loss ?? "N/A";
  document.getElementById("kpiMeanLatency").innerText = `${summary.mean_latency_ms ?? 0} ms`;

  const runnerCard = document.getElementById("liveRunnerCard");
  runnerCard.hidden = !liveApiAvailable;
  const cfgSel = document.getElementById("expConfigSelect");
  if (cfgSel && cfgSel.options.length === 0) {
    for (const key of (expPayload.available_configs || [])) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = key;
      cfgSel.appendChild(opt);
    }
  }

  renderJob178Card(summary.job178 || {});
  renderBars();
  renderCompare();
  renderTable();
}

function renderJob178Card(j) {
  const el = document.getElementById("job178Details");
  const rows = [
    ["Job ID", `Slurm #${j.job_id || "178"}`],
    ["Base Architecture", j.base_model || "Qwen2.5-7B-Instruct"],
    ["LoRA Config", `r=${j.lora_r || 16}, alpha=${j.lora_alpha || 32}`],
    ["Optimization Steps", j.steps ?? 100],
    ["Loss Trajectory", j.first_loss && j.last_loss ? `${j.first_loss} -> ${j.last_loss}` : "N/A"],
    ["Wall-Clock Runtime", `${j.elapsed_seconds ?? 26.07} s (L40S)`],
  ];
  el.innerHTML = rows.map(([k, v]) => `
    <div class="kv-item">
      <dt>${escapeHtml(k)}</dt>
      <dd>${escapeHtml(v)}</dd>
    </div>
  `).join("");
}

function renderBars() {
  const container = document.getElementById("expBarsContainer");
  const chartRuns = (expPayload.runs || []).filter(r => r.kind !== "Live Verification").slice(0, 10);
  container.innerHTML = chartRuns.map((r, idx) => {
    const progId = `prog-${idx}`;
    const em = Number(r.exact_match_pct ?? 0);
    return `
      <div class="bar-row">
        <div class="bar-meta">
          <label for="${progId}"><strong>${escapeHtml(r.run_id)}</strong> · ${escapeHtml(r.name)}</label>
          <span><strong>${escapeHtml(em)}%</strong> EM</span>
        </div>
        <progress id="${progId}" class="exp-progress" max="100" value="${escapeHtml(em)}"></progress>
      </div>
    `;
  }).join("");
}

function renderCompare() {
  const card = document.getElementById("expCompareCard");
  const body = document.getElementById("expCompareBody");
  if (selectedRunIds.length !== 2) {
    card.hidden = true;
    return;
  }
  const runs = expPayload.runs || [];
  const a = runs.find(r => r.run_id === selectedRunIds[0]);
  const b = runs.find(r => r.run_id === selectedRunIds[1]);
  if (!a || !b) {
    card.hidden = true;
    return;
  }
  const emDelta = (Number(b.exact_match_pct || 0) - Number(a.exact_match_pct || 0)).toFixed(1);
  const latDelta = (Number(b.latency_ms || 0) - Number(a.latency_ms || 0)).toFixed(1);
  card.hidden = false;
  body.innerHTML = [a, b].map((r, idx) => `
    <div class="compare-panel">
      <strong>${idx === 0 ? "Baseline" : "Candidate"}: ${escapeHtml(r.run_id)}</strong>
      <div>${escapeHtml(r.name)} (${escapeHtml(r.kind)})</div>
      <div><code>${escapeHtml(r.hyperparams)}</code></div>
      <div>Exact Match: <strong>${escapeHtml(r.exact_match_pct)}%</strong> • Doc ID: <strong>${escapeHtml(r.doc_id_acc_pct ?? "N/A")}%</strong> • Latency: <strong>${escapeHtml(r.latency_ms)} ms</strong></div>
      ${idx === 1 ? `<div>Delta vs ${escapeHtml(a.run_id)}: <strong>${Number(emDelta) >= 0 ? "+" : ""}${escapeHtml(emDelta)}% EM</strong>, <strong>${Number(latDelta) >= 0 ? "+" : ""}${escapeHtml(latDelta)} ms</strong></div>` : ""}
    </div>
  `).join("");
}

function renderTable() {
  const tbody = document.getElementById("expTableBody");
  const query = (document.getElementById("expSearchInput")?.value || "").trim().toLowerCase();
  const filtered = (expPayload.runs || []).filter(r => {
    const kindOk = activeKind === "all" || r.kind === activeKind;
    if (!kindOk) return false;
    if (!query) return true;
    const blob = `${r.run_id} ${r.kind} ${r.name} ${r.model} ${r.hyperparams} ${r.status}`.toLowerCase();
    return blob.includes(query);
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11">No matching experiment runs found.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(r => {
    const checked = selectedRunIds.includes(r.run_id);
    const isBest = Number(r.exact_match_pct) >= 98.4;
    const isOk = ["COMPLETED", "VERIFIED", "PRIMARY", "RECOMMENDED", "AUTO_APPROVED", "DOCUMENT_VERIFIED_WITH_RAG", "ANSWERED_WITH_RAG"].includes(r.status);
    return `
      <tr class="${isBest ? "is-best" : ""}">
        <td><input type="checkbox" class="cmp-check" data-run-id="${escapeHtml(r.run_id)}" aria-label="Compare ${escapeHtml(r.run_id)}" ${checked ? "checked" : ""}></td>
        <th scope="row">${escapeHtml(r.run_id)}</th>
        <td>${escapeHtml(r.kind)}</td>
        <td>${escapeHtml(r.name)}</td>
        <td><code>${escapeHtml(r.hyperparams)}</code></td>
        <td><strong>${escapeHtml(r.exact_match_pct)}%</strong></td>
        <td>${r.doc_id_acc_pct !== null && r.doc_id_acc_pct !== undefined ? escapeHtml(r.doc_id_acc_pct) + "%" : "N/A"}</td>
        <td>${escapeHtml(r.hallucination_rate_pct ?? 0)}%</td>
        <td>${escapeHtml(r.latency_ms)} ms</td>
        <td>${escapeHtml(r.loss_summary || "-")}</td>
        <td><span class="status-pill ${isOk ? "ok" : ""}">${escapeHtml(r.status)}</span></td>
      </tr>
    `;
  }).join("");

  tbody.querySelectorAll(".cmp-check").forEach(cb => {
    cb.addEventListener("change", () => {
      const id = cb.dataset.runId;
      if (cb.checked) {
        if (!selectedRunIds.includes(id)) {
          if (selectedRunIds.length >= 2) selectedRunIds.shift();
          selectedRunIds.push(id);
        }
      } else {
        selectedRunIds = selectedRunIds.filter(x => x !== id);
      }
      renderCompare();
      renderTable();
    });
  });
}
