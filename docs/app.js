let expPayload = null;

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));
}

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

function initDashboard() {
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

  loadExperiments();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initDashboard);
} else {
  initDashboard();
}

async function loadExperiments() {
  const errEl = document.getElementById("expLoadError");
  if (typeof window !== "undefined" && window.FINKYC_SNAPSHOT) {
    expPayload = window.FINKYC_SNAPSHOT;
    if (errEl) errEl.hidden = true;
    renderAll();
  }
  try {
    const res = await fetch("./experiments.json?v=20261126b", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    expPayload = await res.json();
    if (errEl) errEl.hidden = true;
    renderAll();
  } catch (err) {
    if (!expPayload) {
      if (errEl) {
        errEl.innerText = `Unable to load static snapshot ./experiments.json (${err.message}). Open via GitHub Pages or an HTTP server.`;
        errEl.hidden = false;
      }
      const tbody = document.getElementById("expTableBody");
      if (tbody) tbody.innerHTML = `<tr><td colspan="10">Failed to load static evaluation data.</td></tr>`;
    }
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

  renderJob178Card(summary.job178 || {});
  renderBars();
  renderTable();
  renderBenchmarks();
  renderModelCard();
}

function renderJob178Card(j) {
  const el = document.getElementById("job178Details");
  if (!el) return;
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
  if (!container) return;
  const runs = expPayload.runs || [];
  container.innerHTML = runs.map((r, idx) => {
    const progId = `prog-${idx}`;
    const em = Number(r.exact_match_pct ?? 0);
    return `
      <div class="bar-row">
        <div class="bar-meta">
          <label for="${progId}"><strong>${escapeHtml(r.run_id)}</strong> &middot; ${escapeHtml(r.name)}</label>
          <span><strong>${escapeHtml(em)}%</strong> EM</span>
        </div>
        <progress id="${progId}" class="exp-progress" max="100" value="${escapeHtml(em)}"></progress>
      </div>
    `;
  }).join("");
}

function renderTable() {
  const tbody = document.getElementById("expTableBody");
  if (!tbody) return;
  const runs = expPayload.runs || [];
  if (runs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10">No evaluation runs found.</td></tr>`;
    return;
  }

  tbody.innerHTML = runs.map(r => {
    const isBest = Number(r.exact_match_pct) >= 98.4;
    const isOk = ["COMPLETED", "VERIFIED", "PRIMARY", "RECOMMENDED"].includes(r.status);
    return `
      <tr class="${isBest ? "is-best" : ""}">
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
}

function renderBenchmarks() {
  const fourWayBody = document.getElementById("fourWayTableBody");
  if (fourWayBody) {
    const entries = Object.entries(expPayload.four_way_comparison || {});
    fourWayBody.innerHTML = entries.length === 0
      ? `<tr><td colspan="8">No 4-way benchmark data available.</td></tr>`
      : entries.map(([key, row]) => {
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

  const benchBody = document.getElementById("benchmarkTableBody");
  if (benchBody) {
    const entries = Object.entries(expPayload.benchmark_matrix || {}).filter(([k]) => !k.startsWith("_"));
    benchBody.innerHTML = entries.length === 0
      ? `<tr><td colspan="6">No ablation benchmark data available.</td></tr>`
      : entries.map(([key, row]) => {
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
