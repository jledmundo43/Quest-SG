// 投资闯关 SG - 前端。游戏状态保存在 localStorage；计算由服务器完成。
const KEY = "investquest_zh_state_v1";
const CHAT_KEY = "investquest_zh_chat_v1";
let state = null, V = null, clsFilter = "全部", chat = [], guideEnabled = null, busy = false;

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) => "SGD " + Number(n).toLocaleString("en-SG", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt0 = (n) => fmt(n, 0);
const sign = (n) => (n >= 0 ? "+" : "-");
const pnlCls = (n) => (n > 0.005 ? "up" : n < -0.005 ? "down" : "");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || "出了点问题，请再试一次。");
  return j;
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.remove("hidden"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), 2600); }
function apply(j) { state = j.state; V = j.view; save(); render(); }

async function newGame() {
  apply(await api("/api/new"));
  chat = []; localStorage.removeItem(CHAT_KEY); renderChat();
  $("welcome").classList.remove("hidden");
}

async function init() {
  fetch("/api/health").then((r) => r.json()).then((h) => { guideEnabled = h.guide_enabled; $("guideOff").classList.toggle("hidden", !!guideEnabled); }).catch(() => {});
  try { chat = JSON.parse(localStorage.getItem(CHAT_KEY) || "[]"); } catch (e) { chat = []; }
  renderChat();
  const saved = localStorage.getItem(KEY);
  if (saved) {
    try { apply(await api("/api/view", { state: JSON.parse(saved) })); return; } catch (e) { /* fall through */ }
  }
  await newGame();
}

function render() {
  $("date").textContent = V.date;
  $("total").textContent = fmt(V.total);
  $("gain").textContent = `${sign(V.gain)}${fmt(Math.abs(V.gain))} (${sign(V.gain)}${Math.abs(V.gain / V.start * 100).toFixed(1)}%)`;
  $("cash").textContent = fmt(V.cash) + (V.cash < 0 ? "（欠款）" : ""); $("cash").className = V.cash < 0 ? "down" : "";
  $("invested").textContent = fmt(V.invested_value);
  $("total2").textContent = fmt(V.total);
  $("real").textContent = fmt(V.real_total);
  const m = V.market;
  $("mkt").innerHTML = [`STI ${m.sti.toLocaleString()}`, `新元利率 ${m.rate}%`, `10年期国债收益率 ${m.y10}%`, `通胀 ${m.infl}%`, `美元/新元 ${m.usd}`, `欧元/新元 ${m.eur}`].map((s) => `<span>${s}</span>`).join("");
  const eb = $("eventsBar");
  if (m.events.length) { eb.classList.remove("hidden"); eb.innerHTML = "⚠️ 正在发生：" + m.events.map((e) => `<b>${esc(e.label)}</b>`).join("、"); }
  else eb.classList.add("hidden");
  renderPositions(); renderProducts(); renderLog(); drawChart();
}

function renderPositions() {
  const el = $("positions");
  if (!V.positions.length) { el.innerHTML = `<div class="empty">你现在只有现金。银行活期几乎不赚钱（年化约 0.05%）。去 <b>投资</b> 页买点东西吧。</div>`; return; }
  el.innerHTML = V.positions.map((p) => `
    <div class="pos">
      <div class="pos-top">
        <div><div class="pos-name">${esc(p.name)}</div><div class="pos-sub">${esc(p.cls)} · 风险 ${p.risk}/5</div></div>
        <div class="pos-r"><b>${fmt(p.value)}</b><div class="${pnlCls(p.pnl)}">${sign(p.pnl)}${fmt(Math.abs(p.pnl))} (${p.pnl_pct.toFixed(1)}%)</div></div>
      </div>
      <div class="pos-sub">${esc(p.details)}</div>
      <div class="pos-sub">成本 ${fmt(p.cost)}${p.income ? ` · 已收收益 ${fmt(p.income)}` : ""}</div>
      <div style="margin-top:6px"><button class="btn small ghost" onclick="openSell(${p.id})">卖出</button></div>
    </div>`).join("");
}

function renderProducts() {
  const chips = ["全部", ...V.classes];
  $("classChips").innerHTML = chips.map((c) => `<button class="chip ${c === clsFilter ? "on" : ""}" onclick="setCls('${c}')">${c}</button>`).join("");
  let html = "";
  for (const c of V.classes) {
    if (clsFilter !== "全部" && clsFilter !== c) continue;
    html += `<div class="cls-h">${c}</div>`;
    for (const p of V.products.filter((x) => x.cls === c)) {
      html += `<div class="prod" onclick="openBuy('${p.id}')"><div class="prod-top"><div class="prod-n">${esc(p.name)}</div><span class="risk r${p.risk}">风险 ${p.risk}</span></div><div class="prod-q">${esc(p.quote)}</div></div>`;
    }
  }
  $("products").innerHTML = html;
}
window.setCls = (c) => { clsFilter = c; renderProducts(); };

function renderLog() {
  $("log").innerHTML = V.log.slice().reverse().map((e) => `
    <div class="log-e ${e.kind}"><div class="log-d">${e.date}</div>${e.headline ? `<div class="log-h">${esc(e.headline)}</div>` : ""}<div>${esc(e.text)}</div></div>`).join("");
}

function drawChart() {
  const cv = $("chart"), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 320, h = 180;
  cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const H = V.history; const vals = H.map((x) => x.total);
  let lo = Math.min(...vals, V.start), hi = Math.max(...vals, V.start);
  const pad = (hi - lo) * 0.1 || V.start * 0.05; lo -= pad; hi += pad;
  const L = 52, R = 8, T = 8, B = 22;
  const X = (i) => L + (H.length > 1 ? (i / (H.length - 1)) * (w - L - R) : (w - L - R) / 2);
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (h - T - B);
  g.font = "11px sans-serif"; g.fillStyle = "#6b7785"; g.strokeStyle = "#e3e8ee"; g.lineWidth = 1;
  for (let k = 0; k <= 3; k++) {
    const v = lo + (hi - lo) * k / 3, y = Y(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
    g.fillText(v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : (v / 1000).toFixed(0) + "k", 4, y + 4);
  }
  g.setLineDash([4, 4]); g.strokeStyle = "#94a3b8"; g.beginPath(); g.moveTo(L, Y(V.start)); g.lineTo(w - R, Y(V.start)); g.stroke(); g.setLineDash([]);
  g.fillText(H[0].date, L, h - 6); const last = H[H.length - 1].date; g.fillText(last, w - R - g.measureText(last).width, h - 6);
  if (H.length > 1) {
    const grad = g.createLinearGradient(0, T, 0, h - B); grad.addColorStop(0, "rgba(20,184,166,.35)"); grad.addColorStop(1, "rgba(20,184,166,0)");
    g.beginPath(); H.forEach((p, i) => (i ? g.lineTo(X(i), Y(p.total)) : g.moveTo(X(i), Y(p.total))));
    g.lineTo(X(H.length - 1), h - B); g.lineTo(X(0), h - B); g.closePath(); g.fillStyle = grad; g.fill();
    g.beginPath(); H.forEach((p, i) => (i ? g.lineTo(X(i), Y(p.total)) : g.moveTo(X(i), Y(p.total))));
    g.strokeStyle = "#0f766e"; g.lineWidth = 2; g.stroke();
  } else { g.fillStyle = "#0f766e"; g.beginPath(); g.arc(X(0), Y(vals[0]), 4, 0, 7); g.fill(); }
}
window.addEventListener("resize", () => V && drawChart());

function openModal(id) { $(id).classList.remove("hidden"); }
function closeModals() { document.querySelectorAll(".modal:not(#welcome)").forEach((m) => m.classList.add("hidden")); }
document.querySelectorAll("[data-close]").forEach((b) => (b.onclick = closeModals));
document.querySelectorAll(".modal").forEach((m) => m.addEventListener("click", (e) => { if (e.target === m && m.id !== "welcome") closeModals(); }));

window.openBuy = (pid) => {
  const p = V.products.find((x) => x.id === pid);
  const costs = [];
  if (p.buy_cost) costs.push(`买入费用 ${(p.buy_cost * 100).toFixed(2).replace(/\.?0+$/, "")}%`);
  if (p.sell_cost) costs.push(`卖出费用 ${(p.sell_cost * 100).toFixed(2).replace(/\.?0+$/, "")}%`);
  $("tradeBody").innerHTML = `
    <h3>${esc(p.name)}</h3>
    <div><span class="risk r${p.risk}">风险 ${p.risk}/5</span> <span class="small-mut">${esc(p.cls)}</span></div>
    <p class="desc">${esc(p.desc)}</p>
    <div class="small-mut">${esc(p.quote)}</div>
    <div class="small-mut">最低 ${fmt0(p.min)}${costs.length ? " · " + costs.join(" · ") : ""}</div>
    <div class="field"><label class="small-mut">投资金额（可用现金 ${fmt(V.cash)}）</label>
      <input id="amt" type="number" inputmode="decimal" min="0" step="100" placeholder="例如 5000">
      <div class="quick">${[1000, 5000, 10000].map((a) => `<button class="btn small ghost" onclick="$('amt').value=${a}">${a.toLocaleString()}</button>`).join("")}
        <button class="btn small ghost" onclick="$('amt').value=${Math.max(0, Math.floor(V.cash * 100) / 100)}">全部现金</button></div></div>
    <div class="err" id="terr"></div>
    <button class="btn wide" id="doBuy">买入</button>`;
  $("doBuy").onclick = () => trade({ action: "buy", product_id: pid, amount: parseFloat($("amt").value) });
  openModal("tradeModal");
};

window.openSell = (id) => {
  const p = V.positions.find((x) => x.id === id);
  $("tradeBody").innerHTML = `
    <h3>卖出 ${esc(p.name)}</h3>
    <p class="desc">当前市值 <b>${fmt(p.value)}</b> · 盈亏 <span class="${pnlCls(p.pnl)}">${sign(p.pnl)}${fmt(Math.abs(p.pnl))}</span></p>
    <div class="small-mut">${esc(p.details)}</div>
    ${p.sell_cost ? `<div class="small-mut">卖出费用 ${(p.sell_cost * 100).toFixed(2).replace(/\.?0+$/, "")}%（成交价会略低于市价）。</div>` : ""}
    <div class="field"><label class="small-mut">卖出金额（新加坡元；留空表示全部卖出）</label>
      <input id="amt" type="number" inputmode="decimal" min="0" step="100" placeholder="全部"></div>
    <div class="err" id="terr"></div>
    <button class="btn wide" id="doSell">卖出</button>`;
  $("doSell").onclick = () => { const a = parseFloat($("amt").value); trade({ action: "sell", position_id: id, amount: isNaN(a) ? null : a }); };
  openModal("tradeModal");
};

async function trade(body) {
  if (busy) return; busy = true;
  try { apply(await api("/api/trade", { state, ...body })); closeModals(); toast(body.action === "buy" ? "成交！去「资产」页看看吧。" : "已卖出，现金已入账。"); }
  catch (e) { $("terr").textContent = e.message; }
  finally { busy = false; }
}

document.querySelectorAll(".btn.time").forEach((b) => (b.onclick = () => advance(parseInt(b.dataset.m))));
async function advance(months) {
  if (busy) return; busy = true;
  document.querySelectorAll(".btn.time").forEach((b) => (b.disabled = true));
  try { const j = await api("/api/advance", { state, months }); apply(j); showJump(j.summary); }
  catch (e) { toast(e.message); }
  finally { busy = false; document.querySelectorAll(".btn.time").forEach((b) => (b.disabled = false)); }
}

function showJump(s) {
  const evs = s.events.length ? s.events.map((e) => `<div class="jump-ev ${e.kind}"><div class="log-d">${e.date}</div>${e.headline ? `<b>${esc(e.headline)}</b><br>` : ""}${esc(e.text)}</div>`).join("")
    : `<div class="small-mut">这段时间比较平静，没有重大市场事件。</div>`;
  const pos = s.positions.length ? s.positions.map((p) => `<div class="pos-top" style="font-size:14px;padding:3px 0"><span>${esc(p.name)}</span><span class="${pnlCls(p.change)}">${sign(p.change)}${fmt(Math.abs(p.change))}</span></div>`).join("") : "";
  $("jumpBody").innerHTML = `
    <h3>${s.from_date} → ${s.to_date}</h3>
    <div class="small-mut">你的总资产</div>
    <div class="big ${pnlCls(s.change)}">${fmt(s.end_total)}</div>
    <div class="${pnlCls(s.change)}">${sign(s.change)}${fmt(Math.abs(s.change))}（${s.change_pct >= 0 ? "+" : ""}${s.change_pct}%），历时 ${s.months} 个月</div>
    <div class="small-mut" style="margin-top:4px">STI 指数 ${s.sti_change_pct >= 0 ? "+" : ""}${s.sti_change_pct}% · 新元利率 ${s.rate_from}% → ${s.rate_to}% · 当前通胀 ${s.infl}%</div>
    ${pos ? `<div class="card-h" style="margin-top:12px">各持仓变动（含收益）</div>${pos}` : ""}
    <div class="card-h" style="margin-top:12px">发生了什么</div>${evs}
    <button class="btn ghost wide" id="explainBtn" style="margin-top:12px">💬 问问向导怎么回事</button>
    <div id="explainOut"></div>
    <button class="btn wide" data-close2>继续</button>`;
  $("jumpBody").querySelector("[data-close2]").onclick = closeModals;
  $("explainBtn").onclick = async () => {
    $("explainBtn").disabled = true; $("explainOut").innerHTML = `<div class="explain">思考中…</div>`;
    try { const j = await api("/api/explain", { state, summary: s }); $("explainOut").innerHTML = `<div class="explain">${esc(j.reply)}</div>`; }
    catch (e) { $("explainOut").innerHTML = `<div class="explain">${esc(e.message)}</div>`; }
  };
  openModal("jumpModal");
}

function renderChat() {
  const el = $("chatLog");
  el.innerHTML = `<div class="msg bot">嗨！有什么想问的尽管问，比如「什么是债券？」或「为什么我的 ETF 跌了？」</div>` +
    chat.map((m) => `<div class="msg ${m.role === "user" ? "me" : "bot"}">${esc(m.content)}</div>`).join("");
  el.scrollTop = el.scrollHeight;
}
$("chatForm").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("chatInput").value.trim(); if (!q) return;
  $("chatInput").value = ""; chat.push({ role: "user", content: q }); renderChat();
  $("chatLog").insertAdjacentHTML("beforeend", `<div class="msg bot" id="typing">…</div>`);
  try { const j = await api("/api/chat", { state, messages: chat }); chat.push({ role: "assistant", content: j.reply }); }
  catch (err) { chat.push({ role: "assistant", content: err.message }); }
  chat = chat.slice(-30); localStorage.setItem(CHAT_KEY, JSON.stringify(chat)); renderChat();
};

document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  window.scrollTo(0, 0); if (b.dataset.tab === "portfolio") drawChart();
}));
$("startBtn").onclick = () => $("welcome").classList.add("hidden");
$("resetBtn").onclick = () => { if (confirm("确定新开一局？当前持仓会被清空。")) newGame(); };

init();
