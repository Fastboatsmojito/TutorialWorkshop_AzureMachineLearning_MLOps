const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const traceEl = document.getElementById("tool-trace");
const suggestions = document.getElementById("suggestions");

let history = [];
let chart = null;

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = content;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    if (j.aoai_configured && j.aml_configured) {
      statusEl.textContent = `online · ${j.model || "model"}`;
      statusEl.className = "status ok";
    } else {
      statusEl.textContent = "partially configured";
      statusEl.className = "status bad";
    }
  } catch {
    statusEl.textContent = "offline";
    statusEl.className = "status bad";
  }
}

function renderChart(chartData) {
  if (!chartData) return;
  summaryEl.innerHTML = "";
  const s = chartData.summary || {};
  summaryEl.innerHTML =
    `Avg <b>$${s.avg ?? "?"}</b> · Min <b>$${s.min ?? "?"}</b> · Max <b>$${s.max ?? "?"}</b> ` +
    `${chartData.units || ""} · Peak <b>${(s.peak_hour || "").replace("T", " ")}</b>`;

  const labels = chartData.labels.map((t) => t.replace("T", " ").slice(5, 16));
  const ctx = document.getElementById("chart");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `Forecast price (${chartData.units || "CAD/MWh"})`,
          data: chartData.prices,
          borderColor: "#f5a623",
          backgroundColor: "rgba(245,166,35,0.15)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#93a4b7" } } },
      scales: {
        x: { ticks: { color: "#93a4b7", maxTicksLimit: 8 }, grid: { color: "#22344733" } },
        y: { ticks: { color: "#93a4b7" }, grid: { color: "#22344733" } },
      },
    },
  });
}

function renderTrace(trace, guardrails) {
  let html = "";
  if (trace && trace.length) {
    html +=
      "tools called: " +
      trace.map((t) => `<span class="pill">${t.tool}(${Object.keys(t.args || {}).join(", ")})</span>`).join("");
  }
  if (guardrails && guardrails.length) {
    const pills = guardrails
      .map((g) => {
        const blocked = g.blocked === true;
        const cls = blocked ? "pill blocked" : "pill ok";
        const mark = blocked ? "⛔" : "✓";
        return `<span class="${cls}">${mark} ${g.guardrail}</span>`;
      })
      .join("");
    html += `${html ? "<br>" : ""}guardrails: ${pills}`;
  }
  traceEl.innerHTML = html;
}

async function send(text) {
  if (!text.trim()) return;
  addMessage("user", text);
  history.push({ role: "user", content: text });
  input.value = "";
  sendBtn.disabled = true;
  const typing = addMessage("assistant typing", "thinking…");

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    const j = await r.json();
    typing.remove();
    addMessage("assistant", j.reply || "(no response)");
    history.push({ role: "assistant", content: j.reply || "" });
    renderChart(j.chart);
    renderTrace(j.tool_trace, j.guardrails);
  } catch (e) {
    typing.remove();
    addMessage("assistant", "Network error: " + e.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
suggestions.addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") send(e.target.textContent);
});

checkHealth();
addMessage("assistant", "Hi — I'm your ETRM forecast assistant for the Alberta (AESO) market. Ask me for a price forecast, the model's accuracy, or run a what-if scenario.");
