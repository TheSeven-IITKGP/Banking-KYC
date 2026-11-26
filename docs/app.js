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

function wireTabList(tablistId, onActivate) {
  const tablist = document.getElementById(tablistId);
  if (!tablist) return;
  const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
  const selectTab = (tab, moveFocus) => {
    tabs.forEach(t => {
      const selected = t === tab;
      t.setAttribute("aria-selected", String(selected));
      t.setAttribute("tabindex", selected ? "0" : "-1");
    });
    if (moveFocus) tab.focus();
    onActivate(tab);
  };

  tabs.forEach((tab, idx) => {
    tab.addEventListener("click", () => selectTab(tab, false));
    tab.addEventListener("keydown", event => {
      let nextIdx = -1;
      if (event.key === "ArrowRight") nextIdx = (idx + 1) % tabs.length;
      else if (event.key === "ArrowLeft") nextIdx = (idx - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") nextIdx = 0;
      else if (event.key === "End") nextIdx = tabs.length - 1;
      if (nextIdx >= 0) {
        event.preventDefault();
        selectTab(tabs[nextIdx], true);
      }
    });
  });
}

function bindControls() {
  document.getElementById("expSearchInput")?.addEventListener("input", renderTable);
  document.getElementById("runExperimentBtn")?.addEventListener("click", runLiveExperiment);
  document.getElementById("clearCompareBtn")?.addEventListener("click", () => {
    selectedRunIds = [];
    renderCompare();
    renderTable();
  });

  wireTabList("pageTablist", tab => {
    const targetId = tab.dataset.panel;
    document.querySelectorAll(".page-tabpanel").forEach(panel => {
      panel.hidden = panel.id !== targetId;
    });
  });

  wireTabList("benchSubTablist", tab => {
    const targetId = tab.dataset.subpanel;
    ["viewFourWayTable", "viewAblationTable"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.hidden = id !== targetId;
    });
  });

  wireTabList("kindTablist", tab => {
    activeKind = tab.dataset.kind || "all";
    renderTable();
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
  renderBenchmarks();
  renderModelCard();
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

function renderBenchmarks() {
  const fourWayBody = document.getElementById("fourWayTableBody");
  if (fourWayBody) {
    const entries = Object.entries(expPayload.four_way_comparison || {});
    if (entries.length === 0) {
      fourWayBody.innerHTML = `<tr><td colspan="8">No 4-way benchmark data available.</td></tr>`;
    } else {
      fourWayBody.innerHTML = entries.map(([key, row]) => {
        const isBest = Boolean(row.setup_highlight || key.includes("Fine-Tuned"));
        return `
          <tr class="${isBest ? "is-best" : ""}">
            <th scope="row">${escapeHtml(key)}<span class="setup-subname">${escapeHtml(row.setup_name)}</span></th>
            <td><strong>${escapeHtml(row.exact_match_pct)}%</strong></td>
            <td>${escapeHtml(row.document_number_acc)}%</td>
            <td>${row.citation_precision_pct > 0 ? escapeHtml(row.citation_precision_pct) + "%" : "N/A"}</td>
            <td>${escapeHtml(row.hallucination_rate_pct)}%</td>
            <td>${escapeHtml(row.mean_latency_ms)} ms</td>
            <td>${escapeHtml(row.judge_faithfulness)} / 5</td>
            <td><span class="status-pill ${isBest ? "ok" : ""}">${escapeHtml(row.judge_composite_score)} / 10</span></td>
          </tr>
        `;
      }).join("");
    }
  }

  const benchBody = document.getElementById("benchmarkTableBody");
  if (benchBody) {
    const entries = Object.entries(expPayload.benchmark_matrix || {}).filter(([k]) => !k.startsWith("_"));
    if (entries.length === 0) {
      benchBody.innerHTML = `<tr><td colspan="6">No ablation benchmark data available.</td></tr>`;
    } else {
      benchBody.innerHTML = entries.map(([key, row]) => {
        const isBest = key.includes("Config 1");
        return `
          <tr class="${isBest ? "is-best" : ""}">
            <th scope="row">${escapeHtml(key)}</th>
            <td><strong>${escapeHtml(row.exact_match_pct)}%</strong></td>
            <td>${escapeHtml(row.document_number_acc)}%</td>
            <td>${escapeHtml(row.name_f1)}</td>
            <td>${escapeHtml(row.valid_json_pct)}%</td>
            <td>${escapeHtml(row.mean_latency_ms)} ms</td>
          </tr>
        `;
      }).join("");
    }
  }
}

function renderModelCard() {
  const card = expPayload.model_card || {};
  const ts = card.training_summary || {};
  const ac = card.adapter_config || {};
  const summaryEl = document.getElementById("modelCardSummary");
  if (summaryEl) {
    const items = [
      ["Model Name", card.model_name || "FinKYC-Qwen2.5-7B-QLoRA"],
      ["Base Architecture", card.base_model || "Qwen/Qwen2.5-7B-Instruct"],
      ["Adapter Path", card.adapter_path || "models/adapters/qwen2.5-7b-qlora"],
      ["LoRA Rank / Alpha", `r=${ac.r ?? 16}, alpha=${ac.lora_alpha ?? 32}`],
      ["Training Loss", ts.first_loss && ts.last_loss ? `${Number(ts.first_loss).toFixed(4)} -> ${Number(ts.last_loss).toFixed(4)}` : "N/A"],
      ["Adapter Weights", card.adapter_weights_present ? "Ready (safetensors)" : "Missing"],
    ];
    summaryEl.innerHTML = items.map(([k, v]) => `
      <div class="metric-card">
        <div class="metric-val metric-val-compact">${escapeHtml(v)}</div>
        <div class="metric-name">${escapeHtml(k)}</div>
      </div>
    `).join("");
  }

  const metaPre = document.getElementById("modelCardMetaPre");
  if (metaPre) {
    metaPre.innerText = JSON.stringify({
      training_summary: ts,
      experiment_metadata: card.experiment_metadata || {},
      adapter_config: ac,
    }, null, 2);
  }
}
