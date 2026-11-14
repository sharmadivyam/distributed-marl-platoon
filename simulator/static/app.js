'use strict';
// Dashboard for the platoon simulator: draws what the server streams over /stream.

const css = getComputedStyle(document.documentElement);
const C = (name) => css.getPropertyValue(name).trim();
const COL = {
  road: '#434950', mark: 'rgba(255,255,255,0.32)', link: 'rgba(255,255,255,0.16)',
  good: C('--good'), warn: C('--warn'), bad: C('--bad'), leader: C('--leader'),
  beacon: C('--msg-beacon'), data: C('--msg-data'),
};
const MSG_STYLE = {
  beacon: { color: COL.beacon, r: 2.6 },
  state: { color: COL.data, r: 3.4 },
  neurcomm: { color: COL.data, r: 5.4 },
};
// one shade per car, front (light) to back (dark), same as the report figures
const CAR_SHADES = ['#94c4df', '#7db8da', '#64a9d3', '#4a98c9', '#3383be', '#1f6eb3', '#0f5aa3', '#08306b'];
const SPAN_MIN = 150, SPAN_MAX = 330;   // metres of road shown behind the leader (fits the platoon)
const CHART_S = 30;    // seconds of history in the gap chart
const CAR_LEN_M = 4.5;
const BRAKE_LIGHT = 0.8;   // m/s^2 of deceleration that switches the brake lights on

const S = { cfg: null, prev: null, cur: null, arrival: 0, hist: [], playing: false, speed: 1, span: 0 };
let ui = [];

const $ = (id) => document.getElementById(id);

function post(cmd, extra) {
  return fetch('/cmd', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(Object.assign({ cmd }, extra || {})),
  });
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function gapColor(h) {
  const d = Math.abs(h - S.cfg.h_star);
  if (h < 10 || d > 6) return COL.bad;
  if (d > 2) return COL.warn;
  return COL.good;
}

// ------------------------------------------------------------------ layout
function buildLanes(cfg) {
  const main = $('lanes');
  main.innerHTML = '';
  ui = cfg.lanes.map((lane) => {
    const sec = document.createElement('section');
    sec.className = 'lane';
    sec.innerHTML = `
      <div class="lane-head"><h2></h2>
        <div class="chips">
          <span class="chip status"></span>
          <span class="chip">avg reward/step <b class="ravg">n/a</b></span>
          <span class="chip">closest gap <b class="mgap">n/a</b></span>
          <span class="chip">V2V traffic <b class="v2v">n/a</b></span>
        </div>
      </div>
      <div class="lane-body"><canvas class="road"></canvas><canvas class="chart"></canvas></div>`;
    sec.querySelector('h2').textContent = lane.label;
    main.appendChild(sec);
    return {
      road: sec.querySelector('.road'), chart: sec.querySelector('.chart'),
      status: sec.querySelector('.status'), ravg: sec.querySelector('.ravg'),
      mgap: sec.querySelector('.mgap'), v2v: sec.querySelector('.v2v'), minGap: Infinity,
    };
  });
}

function fillControls(cfg) {
  const seed = $('seed');
  if (seed.options.length !== cfg.seeds.length) {
    seed.innerHTML = cfg.seeds.map((s) => `<option value="${s}">${s}</option>`).join('');
  }
  seed.value = String(cfg.seed);
  $('scenario').value = cfg.scenario;
  for (const id of ['brake', 'scenario', 'seed']) $(id).disabled = !!cfg.replay;
  $('brake').title = cfg.replay ? 'Recorded run: the leader\'s braking is part of the recording' : 'B';
  $('auto').textContent = cfg.replay ? 'Replay from start' : 'Auto demo';
  $('netpill').textContent = cfg.network ? cfg.network : 'Network: perfect delivery';
}

// ------------------------------------------------------------------ stream
function onConfig(cfg) {
  const rebuild = !S.cfg || S.cfg.lanes.length !== cfg.lanes.length;
  S.cfg = cfg;
  if (rebuild) buildLanes(cfg);
  cfg.lanes.forEach((lane, k) => { ui[k].minGap = Infinity; });
  fillControls(cfg);
  S.hist = cfg.lanes.map(() => []);
  S.prev = S.cur = null;
}

function onFrame(f) {
  if (!S.cfg) return;
  S.prev = S.cur || f;
  S.cur = f;
  S.arrival = performance.now();
  S.playing = f.playing;
  S.speed = f.speed;
  f.lanes.forEach((L, k) => {
    const hist = S.hist[k];
    if (!hist.length || f.t > hist[hist.length - 1].t) hist.push({ t: f.t, h: L.h });
    while (hist.length && f.t - hist[0].t > CHART_S + 1) hist.shift();
    ui[k].minGap = Math.min(ui[k].minGap, ...L.h);
  });
  updateHeader(f);
  updateChips(f);
  ui.forEach((_, k) => drawChart(k));
}

function updateHeader(f) {
  $('clock').innerHTML = `t = <b>${f.t.toFixed(1)} s</b>`;
  const phase = f.phase === 'cruising' ? '' : ` &middot; <b style="color:${f.phase === 'braking' ? 'var(--danger)' : 'var(--ink)'}">${f.phase}</b>`;
  $('leader').innerHTML = `leader <b>${f.vl.toFixed(1)} m/s</b>${phase}`;
  $('play').innerHTML = f.playing ? '&#10073;&#10073; Pause' : '&#9654; Play';
  $('speed').value = String(f.speed);
}

function updateChips(f) {
  const perS = 1 / S.cfg.dt;
  f.lanes.forEach((L, k) => {
    const u = ui[k];
    if (L.status === 'collision') {
      u.status.textContent = `Collision at ${L.ct.toFixed(1)} s`;
      u.status.className = 'chip status bad';
    } else if (L.status === 'ended') {
      u.status.textContent = 'Run ended';
      u.status.className = 'chip status';
    } else {
      u.status.textContent = f.tick === 0 ? 'Ready' : 'No collisions';
      u.status.className = 'chip status ok';
    }
    u.ravg.textContent = L.ravg === null ? 'n/a' : L.ravg.toFixed(1);
    u.mgap.textContent = `${u.minGap.toFixed(1)} m`;
    const lost = L.dropt !== undefined && L.mt + L.dropt > 0
      ? `, ${Math.round((100 * L.dropt) / (L.mt + L.dropt))}% lost` : '';
    u.v2v.textContent = `${Math.round(L.mk * perS)} msg/s, ${(L.bk * perS / 1000).toFixed(2)} KB/s${lost}`;
  });
}

// ------------------------------------------------------------------ drawing
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawCar(ctx, x, y, w, h, fill, label, labelColor, braking) {
  roundRect(ctx, x - w / 2, y - h / 2, w, h, 4);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.fillStyle = 'rgba(0,0,0,0.25)';              // windscreen, shows the driving direction
  ctx.fillRect(x + w / 2 - 6, y - h / 2 + 3, 3, h - 6);
  if (braking) {                                   // brake lights at the rear
    ctx.fillStyle = '#ff2d2d';
    ctx.shadowColor = 'rgba(255,45,45,0.9)';
    ctx.shadowBlur = 8;
    ctx.fillRect(x - w / 2 - 1, y - h / 2 + 2, 3, 5);
    ctx.fillRect(x - w / 2 - 1, y + h / 2 - 7, 3, 5);
    ctx.shadowBlur = 0;
  }
  ctx.fillStyle = labelColor || '#fff';
  ctx.font = '600 10px -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x - 2, y + 0.5);
}

function drawRoad(k, alpha) {
  const { ctx, w, h } = fitCanvas(ui[k].road);
  const L1 = S.cur.lanes[k];
  const L0 = (S.prev && S.prev.lanes[k]) || L1;
  const moving = S.playing && L1.status === 'ok';
  const a = moving ? alpha : 1;
  const mix = (p, q) => p + (q - p) * a;
  const xl = mix(L0.xl, L1.xl);
  const xs = L1.x.map((x, i) => mix(L0.x[i], x));
  const hs = L1.h.map((g, i) => mix(L0.h[i], g));
  const scale = (w - 60) / S.span;
  const leadX = w - 34;
  const X = (rel) => leadX + rel * scale;
  const pos = (id) => (id < 0 ? X(0) : X(xs[id]));
  const yCar = h * 0.48, yFwd = h * 0.22, yBwd = h * 0.72;

  // road with lane markings that move backwards as the platoon drives forwards
  ctx.fillStyle = COL.road;
  ctx.fillRect(0, 0, w, h);
  ctx.save();
  ctx.strokeStyle = COL.mark;
  ctx.lineWidth = 2;
  ctx.setLineDash([6 * scale, 6 * scale]);
  ctx.lineDashOffset = (xl * scale) % (12 * scale);
  for (const y of [5, h - 5]) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  ctx.restore();

  // V2V links: messages to the car behind travel on the upper line, to the car in front on the lower
  const links = S.cfg.lanes[k].links;
  ctx.lineWidth = 1;
  ctx.strokeStyle = COL.link;
  for (const [src, dst] of links) {
    const y = src < dst ? yFwd : yBwd;
    ctx.beginPath(); ctx.moveTo(pos(src), y); ctx.lineTo(pos(dst), y); ctx.stroke();
  }
  if (moving) {
    for (const [src, dst, kind] of links) {
      const st = MSG_STYLE[kind] || MSG_STYLE.beacon;
      const down = src < dst;
      const lift = kind === 'beacon' ? -4 : 3;
      const y = (down ? yFwd : yBwd) + (down ? lift : -lift);
      const x = pos(src) + (pos(dst) - pos(src)) * alpha;
      ctx.fillStyle = st.color;
      ctx.beginPath(); ctx.arc(x, y, st.r, 0, Math.PI * 2); ctx.fill();
    }
  }

  // vehicles
  const carW = Math.max(18, Math.min(28, CAR_LEN_M * scale));
  const carH = 20;
  drawCar(ctx, X(0), yCar, carW, carH, COL.leader, 'L', '#1b1b1a', moving && S.cur.ul < -BRAKE_LIGHT);
  const dead = new Set(L1.dead || []), stale = new Set(L1.stale || []);
  hs.forEach((g, i) => {
    const x = X(xs[i]);
    if (dead.has(i)) {                               // crashed process: no controller any more
      drawCar(ctx, x, yCar, carW, carH, '#6b7078', '\u2715', 'white', false);
      ctx.fillStyle = '#ffb4b4';
      ctx.font = '600 10px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText('crashed', x, yCar - carH / 2 - 6);
      return;
    }
    drawCar(ctx, x, yCar, carW, carH, gapColor(g), String(i + 1), 'white', moving && L1.u[i] < -BRAKE_LIGHT);
    if (stale.has(i)) {                              // using an old or estimated value this step
      ctx.save();
      ctx.strokeStyle = '#ffffff';
      ctx.setLineDash([3, 2]);
      ctx.lineWidth = 1.5;
      roundRect(ctx, x - carW / 2 - 3, yCar - carH / 2 - 3, carW + 6, carH + 6, 5);
      ctx.stroke();
      ctx.restore();
    }
  });
  for (const [src, dst] of (L1.lost || [])) {         // messages lost this step
    const mx = (pos(src) + pos(dst)) / 2, my = src < dst ? yFwd : yBwd;
    ctx.strokeStyle = '#ff4d4d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(mx - 4, my - 4); ctx.lineTo(mx + 4, my + 4);
    ctx.moveTo(mx + 4, my - 4); ctx.lineTo(mx - 4, my + 4);
    ctx.stroke();
  }

  // gap to the vehicle in front, written under each gap
  ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  hs.forEach((g, i) => {
    const front = i === 0 ? X(0) : X(xs[i - 1]);
    ctx.fillStyle = gapColor(g);
    ctx.fillText(`${g.toFixed(1)} m`, (front + X(xs[i])) / 2, h - 11);
  });

  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.textAlign = 'right';
  ctx.fillText('driving direction →', w - 8, 16);
  if (X(xs[xs.length - 1]) < 8) {
    ctx.textAlign = 'left';
    ctx.fillText('← platoon continues', 8, 16);
  }

  if (L1.status === 'collision') {
    ctx.fillStyle = 'rgba(229,72,77,0.30)';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#fff';
    ctx.font = '600 15px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Collision at ${L1.ct.toFixed(1)} s. This platoon has stopped.`, w / 2, h / 2 - 22);
  }
}

function drawChart(k) {
  const { ctx, w, h } = fitCanvas(ui[k].chart);
  ctx.clearRect(0, 0, w, h);
  const P = { l: 28, r: 8, t: 8, b: 8 };
  const hist = S.hist[k] || [];
  const tEnd = hist.length ? hist[hist.length - 1].t : 0;
  const t0 = Math.max(0, tEnd - CHART_S);
  const X = (t) => P.l + ((t - t0) / CHART_S) * (w - P.l - P.r);
  const Y = (g) => P.t + (1 - Math.max(0, Math.min(g, 50)) / 50) * (h - P.t - P.b);

  ctx.fillStyle = 'rgba(229,72,77,0.08)';            // closer than 10 m
  ctx.fillRect(P.l, Y(10), w - P.l - P.r, Y(0) - Y(10));
  ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 1;
  for (const g of [0, 10, 20, 30, 40, 50]) {
    ctx.strokeStyle = '#ebe9e3';
    ctx.beginPath(); ctx.moveTo(P.l, Y(g)); ctx.lineTo(w - P.r, Y(g)); ctx.stroke();
    ctx.fillStyle = '#8a8984';
    ctx.fillText(String(g), P.l - 5, Y(g));
  }
  ctx.save();
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = '#5d5c58';
  ctx.beginPath(); ctx.moveTo(P.l, Y(S.cfg.h_star)); ctx.lineTo(w - P.r, Y(S.cfg.h_star)); ctx.stroke();
  ctx.restore();

  ctx.lineWidth = 1.6;
  const nCars = hist.length ? hist[hist.length - 1].h.length : 0;
  for (let i = 0; i < nCars; i++) {
    ctx.strokeStyle = CAR_SHADES[Math.min(i, CAR_SHADES.length - 1)];
    ctx.beginPath();
    hist.forEach((p, j) => (j ? ctx.lineTo(X(p.t), Y(p.h[i])) : ctx.moveTo(X(p.t), Y(p.h[i]))));
    ctx.stroke();
  }
  ctx.fillStyle = '#5d5c58';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('gap of each car [m], last 30 s', P.l + 4, P.t + 10);
}

function loop() {
  requestAnimationFrame(loop);
  if (!S.cfg || !S.cur || ui.length !== S.cur.lanes.length) return;
  const tickMs = (1000 * S.cfg.dt) / S.speed;
  const alpha = S.playing ? Math.min(1, (performance.now() - S.arrival) / tickMs) : 1;
  // one shared scale for all lanes, so the same gap looks the same length in every platoon
  const longest = Math.max(...S.cur.lanes.map((L) => -L.x[L.x.length - 1]));
  const want = Math.min(SPAN_MAX, Math.max(SPAN_MIN, longest * 1.1 + 20));
  S.span = S.span ? S.span + (want - S.span) * 0.04 : want;
  for (let k = 0; k < ui.length; k++) drawRoad(k, alpha);
}

// ------------------------------------------------------------------ wiring
const es = new EventSource('/stream');
es.addEventListener('config', (e) => onConfig(JSON.parse(e.data)));
es.addEventListener('frame', (e) => onFrame(JSON.parse(e.data)));
es.onopen = () => { $('banner').hidden = true; };
es.onerror = () => { $('banner').hidden = false; };

const resetToSelection = () => post('reset', { seed: +$('seed').value, scenario: $('scenario').value });
$('play').onclick = () => post('toggle');
$('reset').onclick = resetToSelection;
$('seed').onchange = resetToSelection;
$('scenario').onchange = resetToSelection;
$('speed').onchange = () => post('speed', { value: +$('speed').value });
$('brake').onclick = () => post('brake');
$('auto').onclick = () => post('auto', { seed: +$('seed').value });

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.metaKey || e.ctrlKey) return;
  if (e.code === 'Space') { e.preventDefault(); post('toggle'); }
  else if (e.key === 'b' || e.key === 'B') post('brake');
  else if (e.key === 'r' || e.key === 'R') resetToSelection();
});
window.addEventListener('resize', () => ui.forEach((_, k) => S.cfg && drawChart(k)));
requestAnimationFrame(loop);
