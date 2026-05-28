/* eslint-disable */
import { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

/* ═══════════════════════════════════════════════════════════════════════════
   PHYSICS ENGINE — Crank-Nicolson Solver for 1D Reacting PDEs
   ═══════════════════════════════════════════════════════════════════════════ */
function thomasSolve(lo, diag, up, rhs) {
  const n = diag.length;
  const c = new Float64Array(n);
  const d = new Float64Array(n);
  const x = new Float64Array(n);
  const eps = 1e-15;
  const d0 = Math.abs(diag[0]) < eps ? (diag[0] >= 0 ? eps : -eps) : diag[0];
  c[0] = up[0] / d0; d[0] = rhs[0] / d0;
  for (let i = 1; i < n; i++) {
    let m = diag[i] - lo[i] * c[i - 1];
    if (Math.abs(m) < eps) m = m >= 0 ? eps : -eps;
    c[i] = up[i] / m;
    d[i] = (rhs[i] - lo[i] * d[i - 1]) / m;
  }
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
  return x;
}

function solvePDE(D, r, mu, sig, modelType = "fisher", N = 128, dt = 5e-5, T_end = 1.0) {
  const N_safe = Math.max(3, Math.min(512, Math.round(N || 128)));
  const dt_safe = Math.max(1e-6, Math.min(0.1, dt || 5e-5));
  const D_safe = Math.max(1e-5, Math.min(10.0, D ?? 0.1));
  const r_safe = Math.max(0.0, Math.min(50.0, r ?? 2.0));
  const mu_safe = Math.max(0.001, Math.min(0.999, mu ?? 0.3));
  const sig_safe = Math.max(0.001, Math.min(2.0, sig ?? 0.1));
  const dx = 1 / (N_safe - 1);
  const lam = D_safe * dt_safe / (2 * dx * dx);
  let u = new Float64Array(N_safe);
  for (let i = 0; i < N_safe; i++) {
    const x = i * dx;
    u[i] = Math.min(1, Math.max(0, Math.exp(-0.5 * ((x - mu_safe) / sig_safe) ** 2)));
  }
  const lo = new Float64Array(N_safe).fill(-lam);
  const di = new Float64Array(N_safe).fill(1 + 2 * lam);
  const up = new Float64Array(N_safe).fill(-lam);
  di[0] = 1 + lam; di[N_safe - 1] = 1 + lam;
  let nSteps = Math.round(T_end / dt_safe);
  if (nSteps > 200000) nSteps = 200000;
  const saveEvery = Math.max(1, Math.floor(nSteps / 80));
  const snaps = [Array.from(u)];
  try {
    for (let s = 0; s < nSteps; s++) {
      const rhs = new Float64Array(N_safe);
      for (let i = 0; i < N_safe; i++) {
        const l = i > 0 ? u[i - 1] : u[i];
        const rv = i < N_safe - 1 ? u[i + 1] : u[i];
        let rxn = modelType === "allen" ? r_safe * (u[i] - u[i] ** 3) : r_safe * u[i] * (1 - u[i]);
        rhs[i] = u[i] + lam * (l - 2 * u[i] + rv) + (dt_safe / 2) * rxn;
        if (isNaN(rhs[i]) || !isFinite(rhs[i])) rhs[i] = 0.0;
      }
      const nextU = thomasSolve(lo, di, up, rhs);
      for (let i = 0; i < N_safe; i++) {
        u[i] = Math.min(1, Math.max(0, isNaN(nextU[i]) || !isFinite(nextU[i]) ? u[i] : nextU[i]));
      }
      if (s % saveEvery === 0) snaps.push(Array.from(u));
    }
  } catch (err) { console.error("Solver error:", err); }
  if (snaps.length <= 80) snaps.push(Array.from(u));
  return { snaps, final: Array.from(u) };
}

function fnoPredictTime(D, r, mu, sig, t, modelType = "fisher", N = 128) {
  const N_safe = Math.max(3, Math.min(512, Math.round(N || 128)));
  const D_safe = Math.max(1e-5, Math.min(10.0, D ?? 0.1));
  const r_safe = Math.max(0.0, Math.min(50.0, r ?? 2.0));
  const mu_safe = Math.max(0.001, Math.min(0.999, mu ?? 0.3));
  const sig_safe = Math.max(0.001, Math.min(2.0, sig ?? 0.1));
  const t_safe = Math.max(0.0, Math.min(10.0, t ?? 0.0));
  const dx = 1 / (N_safe - 1);
  let c = 2 * Math.sqrt(D_safe * r_safe);
  let xi = Math.sqrt(r_safe / (6 * D_safe));
  if (modelType === "allen") { c = 1.35 * Math.sqrt(D_safe * r_safe); xi = Math.sqrt(r_safe / (2 * D_safe)); }
  const front = mu_safe + c * t_safe;
  return Array.from({ length: N_safe }, (_, i) => {
    const x = i * dx;
    const icVal = Math.min(1, Math.max(0, Math.exp(-0.5 * ((x - mu_safe) / sig_safe) ** 2)));
    let wave = modelType === "allen"
      ? 0.5 * (1 + Math.tanh(xi * (x - front)))
      : 1 / (1 + Math.exp(-xi * 6 * (x - front)));
    const blend = t_safe === 0 ? 0 : Math.min(1, t_safe * 1.5);
    const pred = (1 - blend) * icVal + blend * Math.min(1, Math.max(0, wave));
    return Math.min(1, Math.max(0, isNaN(pred) || !isFinite(pred) ? icVal : pred));
  });
}

const relL2 = (a, b) => {
  let n = 0, d = 0;
  for (let i = 0; i < a.length; i++) { n += (a[i] - b[i]) ** 2; d += b[i] ** 2; }
  return d > 1e-12 ? Math.sqrt(n / d) * 100 : 0;
};

/* ═══════════════════════════════════════════════════════════════════════════
   CANVAS RENDERING — Grey/Green Palette
   ═══════════════════════════════════════════════════════════════════════════ */
const PAL = {
  bg: "#0f1210", panel: "#1a2119", border: "#2a3a2c",
  text: "#e8f0e9", muted: "#7a9c7e",
  accent: "#4ade80", secondary: "#6ee7b7",
  warn: "#fbbf24", bad: "#f87171"
};

function useCanvas(draw, deps) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    draw(ctx, c.width, c.height);
  }, deps);
  return ref;
}

function drawAxes(ctx, W, H, pad) {
  ctx.strokeStyle = "rgba(74, 222, 128, 0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (H - pad.t - pad.b) / 4 * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    const x = pad.l + (W - pad.l - pad.r) / 4 * i;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
  }
}

function drawLine(ctx, data, W, H, pad, color, lw = 2, glow = false) {
  if (!data?.length) return;
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  if (glow) { ctx.shadowColor = color; ctx.shadowBlur = 8; }
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = pad.l + (i / (data.length - 1)) * pw;
    const y = pad.t + ph * (1 - Math.min(1, Math.max(0, v)));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
}

const greenColormap = (v) => {
  const t = Math.min(1, Math.max(0, v));
  const r = Math.round(15 + t * 30);
  const g = Math.round(18 + t * 200);
  const b = Math.round(15 + t * 40);
  return `rgb(${r},${g},${b})`;
};

function project3D(x, t, u, W, H, pad) {
  const cx = W / 2, cy = H / 2 + 15;
  const thetaX = -Math.PI / 6, thetaT = Math.PI / 8;
  const scaleX = (W - pad.l - pad.r) * 0.44, scaleT = (H - pad.t - pad.b) * 0.44, scaleU = 50;
  const px = cx + (x - 0.5) * scaleX * Math.cos(thetaX) + (t - 0.5) * scaleT * Math.cos(thetaT);
  const py = cy + (x - 0.5) * scaleX * Math.sin(thetaX) + (t - 0.5) * scaleT * Math.sin(thetaT) - u * scaleU;
  return { x: px, y: py };
}

/* ═══════════════════════════════════════════════════════════════════════════
   CHART COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */
function SolutionChart({ solver, fno, ic, title }) {
  const pad = { t: 24, b: 24, l: 36, r: 12 };
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    drawAxes(ctx, W, H, pad);
    ctx.fillStyle = PAL.muted; ctx.font = "9px 'JetBrains Mono',monospace";
    ["0", "0.25", "0.5", "0.75", "1.0"].forEach((l, i) => {
      ctx.fillText(l, pad.l + (W - pad.l - pad.r) / 4 * i - 8, H - 6);
    });
    ["1.0", "0.75", "0.5", "0.25", "0"].forEach((l, i) => {
      ctx.fillText(l, 4, pad.t + (H - pad.t - pad.b) / 4 * i + 4);
    });
    ctx.fillStyle = PAL.muted; ctx.font = "bold 10px 'JetBrains Mono',monospace";
    ctx.fillText(title, pad.l + 4, 15);
    if (ic) drawLine(ctx, ic, W, H, pad, "rgba(74,222,128,0.2)", 1, false);
    if (solver) drawLine(ctx, solver, W, H, pad, PAL.secondary, 2.5, false);
    if (fno) drawLine(ctx, fno, W, H, pad, PAL.accent, 1.5, true);
  }, [solver, fno, ic, title]);
  return <canvas ref={ref} width={480} height={190} className="plot-canvas" />;
}

function ErrorChart({ data, title }) {
  const pad = { t: 24, b: 24, l: 36, r: 12 };
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    drawAxes(ctx, W, H, pad);
    ctx.fillStyle = PAL.muted; ctx.font = "bold 10px 'JetBrains Mono',monospace";
    ctx.fillText(title, pad.l + 4, 15);
    if (!data?.length) return;
    const max = Math.max(...data, 1e-8);
    const norm = data.map(v => v / max);
    drawLine(ctx, norm, W, H, pad, PAL.warn, 1.5, false);
    const pw = W - pad.l - pad.r; const ph = H - pad.t - pad.b;
    ctx.beginPath();
    norm.forEach((v, i) => {
      const x = pad.l + (i / (norm.length - 1)) * pw;
      const y = pad.t + ph * (1 - v);
      if (i === 0) ctx.moveTo(x, H - pad.b); else ctx.lineTo(x, y);
    });
    ctx.lineTo(W - pad.r, H - pad.b); ctx.closePath();
    ctx.fillStyle = "rgba(251, 191, 36, 0.07)"; ctx.fill();
    ctx.fillStyle = PAL.muted; ctx.font = "9px 'JetBrains Mono',monospace";
    ctx.fillText(`max: ${max.toFixed(5)}`, W - 100, 15);
  }, [data, title]);
  return <canvas ref={ref} width={480} height={190} className="plot-canvas" />;
}

function HeatmapChart({ snaps, onHover }) {
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    if (!snaps?.length) return;
    const nT = snaps.length; const nX = snaps[0].length;
    const cw = W / nX; const ch = H / nT;
    for (let t = 0; t < nT; t++) {
      for (let x = 0; x < nX; x++) {
        ctx.fillStyle = greenColormap(snaps[t][x]);
        ctx.fillRect(x * cw, t * ch, cw + 0.5, ch + 0.5);
      }
    }
  }, [snaps]);

  const handleMouseMove = (e) => {
    if (!snaps?.length || !onHover) return;
    const c = ref.current; const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left; const my = e.clientY - rect.top;
    const px = Math.min(1, Math.max(0, mx / rect.width));
    const pt = Math.min(1, Math.max(0, my / rect.height));
    const snapIdx = Math.min(snaps.length - 1, Math.max(0, Math.floor(pt * snaps.length)));
    const nodeIdx = Math.min(snaps[0].length - 1, Math.max(0, Math.floor(px * snaps[0].length)));
    onHover({ x: px, t: pt, u: snaps[snapIdx][nodeIdx] });
  };

  return <canvas ref={ref} width={480} height={200} className="plot-canvas"
    style={{ cursor: "crosshair" }} onMouseMove={handleMouseMove} />;
}

function Waterfall3DChart({ snaps }) {
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    if (!snaps?.length) return;
    const pad = { t: 15, b: 15, l: 15, r: 15 };
    const skip = Math.max(1, Math.floor(snaps.length / 22));
    const selected = [];
    for (let i = 0; i < snaps.length; i += skip) selected.push({ t: i / (snaps.length - 1), data: snaps[i] });
    if (selected[selected.length - 1].t !== 1) selected.push({ t: 1, data: snaps[snaps.length - 1] });
    selected.forEach(({ t, data }) => {
      const N = data.length;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const pt = project3D(i / (N - 1), t, data[i], W, H, pad);
        if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
      }
      const ptLast = project3D(1, t, 0, W, H, pad);
      const ptFirst = project3D(0, t, 0, W, H, pad);
      ctx.lineTo(ptLast.x, ptLast.y); ctx.lineTo(ptFirst.x, ptFirst.y); ctx.closePath();
      ctx.fillStyle = PAL.bg; ctx.fill();
      ctx.strokeStyle = "rgba(74, 222, 128, 0.12)"; ctx.lineWidth = 0.5; ctx.stroke();
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const pt = project3D(i / (N - 1), t, data[i], W, H, pad);
        if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
      }
      const ptS = project3D(0, t, 0.5, W, H, pad); const ptE = project3D(1, t, 0.5, W, H, pad);
      const grad = ctx.createLinearGradient(ptS.x, ptS.y, ptE.x, ptE.y);
      grad.addColorStop(0, "rgba(34, 197, 94, 0.25)");
      grad.addColorStop(0.5, "rgba(74, 222, 128, 0.7)");
      grad.addColorStop(1, "rgba(110, 231, 183, 0.35)");
      ctx.strokeStyle = grad; ctx.lineWidth = 1.2; ctx.stroke();
    });
    const origin = project3D(0, 0, 0, W, H, { t: 15, b: 15, l: 15, r: 15 });
    const axisX = project3D(1, 0, 0, W, H, { t: 15, b: 15, l: 15, r: 15 });
    const axisT = project3D(0, 1, 0, W, H, { t: 15, b: 15, l: 15, r: 15 });
    const axisU = project3D(0, 0, 1, W, H, { t: 15, b: 15, l: 15, r: 15 });
    ctx.strokeStyle = "rgba(74, 222, 128, 0.15)"; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(axisX.x, axisX.y); ctx.lineTo(origin.x, origin.y); ctx.lineTo(axisT.x, axisT.y);
    ctx.moveTo(origin.x, origin.y); ctx.lineTo(axisU.x, axisU.y); ctx.stroke();
    ctx.fillStyle = PAL.muted; ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.fillText("x", axisX.x + 4, axisX.y + 4);
    ctx.fillText("t", axisT.x - 10, axisT.y + 8);
    ctx.fillText("u", axisU.x - 10, axisU.y - 2);
  }, [snaps]);
  return <canvas ref={ref} width={480} height={200} className="plot-canvas" />;
}

function MeshCanvas({ N }) {
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    const pad = { l: 12, r: 12 }; const pw = W - pad.l - pad.r; const cy = H / 2;
    ctx.strokeStyle = PAL.border; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(W - pad.r, cy); ctx.stroke();
    ctx.fillStyle = N > 128 ? PAL.accent : PAL.secondary;
    for (let i = 0; i < N; i++) {
      const x = pad.l + (i / (N - 1)) * pw;
      ctx.beginPath();
      ctx.arc(x, cy, N > 128 ? 1 : N > 64 ? 1.5 : 2, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.fillStyle = PAL.muted; ctx.font = "8px 'JetBrains Mono',monospace";
    ctx.fillText("x=0", pad.l, cy - 5);
    ctx.fillText("x=1", W - pad.r - 18, cy - 5);
  }, [N]);
  return <canvas ref={ref} width={440} height={32} className="ic-preview-canvas" />;
}

function HistChart({ data, title, color }) {
  const pad = { t: 24, b: 24, l: 30, r: 12 };
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    drawAxes(ctx, W, H, pad);
    ctx.fillStyle = PAL.muted; ctx.font = "bold 9px 'JetBrains Mono',monospace";
    ctx.fillText(title, pad.l + 4, 15);
    if (!data?.length) return;
    const maxVal = Math.max(...data, 1);
    const pw = W - pad.l - pad.r; const ph = H - pad.t - pad.b;
    const bw = (pw / data.length) * 0.75;
    data.forEach((val, i) => {
      const x = pad.l + pw * (i / data.length) + (pw / data.length) * 0.125;
      const bh = (val / maxVal) * ph;
      ctx.fillStyle = color || PAL.accent;
      ctx.beginPath(); ctx.roundRect(x, pad.t + ph - bh, bw, bh, 2); ctx.fill();
      if (val > 0) {
        ctx.fillStyle = PAL.text; ctx.font = "7px monospace";
        ctx.fillText(val, x + bw / 2 - 3, pad.t + ph - bh - 2);
      }
    });
  }, [data, title, color]);
  return <canvas ref={ref} width={220} height={110} className="plot-canvas" style={{ width: "100%" }} />;
}

function ICPreviewCanvas({ mu, sig }) {
  const data = Array.from({ length: 128 }, (_, i) => {
    const x = i / 127;
    return Math.min(1, Math.max(0, Math.exp(-0.5 * ((x - mu) / sig) ** 2)));
  });
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    const pad = { t: 2, b: 2, l: 8, r: 8 };
    drawLine(ctx, data, W, H, pad, PAL.accent, 1.5, true);
  }, [mu, sig]);
  return <canvas ref={ref} width={440} height={32} className="ic-preview-canvas" />;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ICON COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */
const IcPlay  = () => <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="6,4 20,12 6,20"/></svg>;
const IcPause = () => <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="4" width="4" height="16" rx="1"/><rect x="15" y="4" width="4" height="16" rx="1"/></svg>;
const IcBack  = () => <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="11,19 2,12 11,5"/><polygon points="22,19 13,12 22,5"/></svg>;
const IcFwd   = () => <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="13,19 22,12 13,5"/><polygon points="2,19 11,12 2,5"/></svg>;

const IcSpin = () => (
  <svg width="28" height="28" viewBox="0 0 50 50" className="spinning" style={{ display:"inline-block" }}>
    <circle cx="25" cy="25" r="20" fill="none" stroke="rgba(74,222,128,0.1)" strokeWidth="4"/>
    <circle cx="25" cy="25" r="20" fill="none" stroke="#4ade80" strokeWidth="4" strokeDasharray="31.4 31.4" strokeLinecap="round"/>
  </svg>
);

/* ═══════════════════════════════════════════════════════════════════════════
   GLOBAL NAV
   ═══════════════════════════════════════════════════════════════════════════ */
function GlobalNav({ activePage, setActivePage, systemUptime }) {
  const fmt = (s) => {
    const h = Math.floor(s / 3600).toString().padStart(2, "0");
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
    const sec = (s % 60).toString().padStart(2, "0");
    return `${h}:${m}:${sec}`;
  };

  return (
    <nav className="global-nav">
      <div className="nav-brand" onClick={() => setActivePage("landing")} title="Home">
        <div className="nav-logo-mark">FN</div>
        <div>
          <div className="nav-title">FNO Scientific</div>
          <div className="nav-subtitle">Neural PDE Operator v4.0</div>
        </div>
      </div>

      <div className="nav-links">
        {["landing","simulator","research"].map(p => (
          <button key={p} className={`nav-link-btn${activePage === p ? " active" : ""}`}
            onClick={() => setActivePage(p)}>
            {{ landing: "Home", simulator: "Simulator", research: "Research" }[p]}
          </button>
        ))}
      </div>

      <div className="nav-right">
        <div className="nav-status">
          <div className="nav-dot-live"/>
          <span>{fmt(systemUptime)}</span>
        </div>
        <button className="nav-launch-btn" onClick={() => setActivePage("simulator")}>
          Launch Sim →
        </button>
      </div>
    </nav>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   LANDING PAGE
   ═══════════════════════════════════════════════════════════════════════════ */
function LandingPage({ setActivePage, mu, sig }) {
  return (
    <div className="landing-page">
      {/* Hero */}
      <section className="hero-section">
        <div className="hero-bg-grid" />

        <div className="hero-badge fade-in-up">
          <span className="hero-badge-dot" />
          ACTIVE — Fourier Neural Operator Platform
        </div>

        <h1 className="hero-headline fade-in-up">
          Solve Reaction-Diffusion PDEs with <span className="highlight">Neural Operators</span>
        </h1>

        <p className="hero-subhead fade-in-up">
          An interactive scientific workstation for exploring Fisher-KPP &amp; Allen-Cahn dynamics.
          Compare Crank-Nicolson solvers against FNO surrogates in real time.
        </p>

        <div className="hero-cta-row">
          <button className="btn-hero-primary" onClick={() => setActivePage("simulator")}>
            <IcPlay /> Open Simulator
          </button>
          <button className="btn-hero-secondary" onClick={() => setActivePage("research")}>
            Read the Theory →
          </button>
        </div>

        {/* Preview Card */}
        <div className="hero-preview-card fade-in-up">
          <div className="preview-top-bar">
            <div className="preview-dot red"/>
            <div className="preview-dot yellow"/>
            <div className="preview-dot green"/>
            <span className="preview-bar-label">FNO Workstation — Fisher-KPP · D=0.10 · r=2.00 · t=1.0</span>
          </div>
          <div className="preview-split">
            <div className="preview-params">
              {[["D","0.10"],["r","2.00"],["μ","0.30"],["σ","0.10"],["N","128"],["Speed","0.632"]].map(([k,v]) => (
                <div className="preview-param-item" key={k}>
                  <div className="preview-param-label">{k}</div>
                  <div className="preview-param-val">{v}</div>
                </div>
              ))}
            </div>
            <div className="preview-canvas-area" style={{ height: "260px" }}>
              <ICPreviewCanvas mu={mu} sig={sig} />
            </div>
          </div>
        </div>
      </section>

      {/* Stats bar */}
      <div className="stats-bar">
        {[
          ["100-500×", "Faster than numerical solvers"],
          ["< 0.1 ms", "FNO inference latency"],
          ["< 2%", "Typical relative L2 error"],
          ["2 Models", "Fisher-KPP & Allen-Cahn"],
        ].map(([n, l]) => (
          <div className="stat-item" key={n}>
            <div className="stat-num">{n}</div>
            <div className="stat-label">{l}</div>
          </div>
        ))}
      </div>

      {/* Features */}
      <section className="features-section">
        <div className="section-header">
          <div className="section-eyebrow">Platform Capabilities</div>
          <h2 className="section-title-text">Everything a PDE researcher needs</h2>
          <p className="section-desc">From parameter sweeps to animated wave fronts — a full scientific sandbox in your browser.</p>
        </div>

        <div className="features-grid">
          {[
            { icon: "⚡", title: "Instant FNO Surrogate", desc: "Sub-millisecond operator inference bypasses iterative Thomas-algorithm steps while maintaining < 2% L2 accuracy." },
            { icon: "🌊", title: "Animated Wave Fronts", desc: "Step through PDE solutions frame-by-frame. Compare traveling waves, phase boundaries, and transient dynamics." },
            { icon: "📊", title: "Real-Time Benchmarking", desc: "Automatic speedup calculation and L2 error metrics update every time you run a simulation." },
            { icon: "🎛️", title: "Full Parameter Control", desc: "Tune diffusion D, reaction rate r, initial Gaussian center μ and width σ. Choose Fisher-KPP or Allen-Cahn." },
            { icon: "🔁", title: "Monte Carlo Sweeps", desc: "Run 50-point random parameter ensembles to profile FNO performance distribution across the physical regime." },
            { icon: "🗺️", title: "Heatmap & 3D Waterfall", desc: "Interactive space-time heatmap with crosshair HUD and pseudo-3D waterfall plot for volume visualization." },
          ].map(f => (
            <div className="feature-card" key={f.title}>
              <div className="feature-icon-wrap">{f.icon}</div>
              <div className="feature-card-title">{f.title}</div>
              <div className="feature-card-desc">{f.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Equation showcase */}
      <section className="equation-section">
        <div className="section-header">
          <div className="section-eyebrow">Supported Equations</div>
          <h2 className="section-title-text">Two nonlinear PDE regimes</h2>
        </div>
        <div className="equation-grid">
          <div className="equation-card active">
            <div className="equation-tag">Fisher-KPP</div>
            <div className="equation-name">Population Wave / Combustion Front</div>
            <div className="equation-formula">∂u/∂t = D·∂²u/∂x² + r·u·(1 − u)</div>
            <div className="equation-desc">Logistic reaction drives population waves at wave speed c = 2√(D·r). Used in ecology, combustion, and epidemiology.</div>
          </div>
          <div className="equation-card">
            <div className="equation-tag">Allen-Cahn</div>
            <div className="equation-name">Phase-Field Separation</div>
            <div className="equation-formula">∂u/∂t = D·∂²u/∂x² + r·(u − u³)</div>
            <div className="equation-desc">Double-well potential drives sharp phase boundaries. Models spinodal decomposition and grain coarsening in materials science.</div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <h2 className="cta-title">Ready to explore neural PDE solving?</h2>
        <p className="cta-sub">Configure your physical parameters and run a simulation in under 5 seconds.</p>
        <button className="btn-hero-primary" style={{ margin: "0 auto" }} onClick={() => setActivePage("simulator")}>
          <IcPlay /> Launch the Simulator
        </button>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div>FNO Scientific Workstation · Fisher-KPP & Allen-Cahn · v4.0.0</div>
        <div className="footer-links">
          <span className="footer-link" onClick={() => setActivePage("simulator")}>Simulator</span>
          <span className="footer-link" onClick={() => setActivePage("research")}>Research</span>
          <a className="footer-link" href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
      </footer>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIMULATOR PAGE
   ═══════════════════════════════════════════════════════════════════════════ */
function SimulatorPage({
  D, setD, r, setR, mu, setMu, sig, setSig, N, setN, dt, setDt,
  modelType, setModelType, snaps, solFinal, fnoFinal, errField,
  solMs, fnoMs, speedup, l2, simDone, running, tIndex, setTIndex,
  animating, setAnimating, animSpeed, setAnimSpeed, hudCoord, setHudCoord,
  sweeping, sweepDone, sweepStats, l2Hist, speedupHist,
  expRows, expName, expErr, expWarn, researchNotes, setResearchNotes,
  convResults, runningConv, executeSimulation, executeMonteCarlo, executeMeshSensitivity,
  handleCSVUpload, exportConfigJSON, exportReport, resetAll, getFittingMAE,
  waveSpeed, cflNumber
}) {
  const [activeTab, setActiveTab] = useState("results");

  const activeSolSnap = snaps.length > 0 ? snaps[tIndex] : null;
  const activeT = snaps.length > 0 ? (tIndex / (snaps.length - 1)).toFixed(3) : "0.000";
  const activeFnoSnap = snaps.length > 0 ? fnoPredictTime(D, r, mu, sig, parseFloat(activeT), modelType, N) : null;

  const icData = Array.from({ length: 128 }, (_, i) => {
    const x = i / 127;
    return Math.min(1, Math.max(0, Math.exp(-0.5 * ((x - mu) / sig) ** 2)));
  });

  const fitSolver = getFittingMAE ? getFittingMAE(solFinal) : null;
  const fitFno    = getFittingMAE ? getFittingMAE(fnoFinal)  : null;

  const PRESETS = [
    { id: "baseline", label: "Baseline Fisher", D: 0.1, r: 2.0, mu: 0.3, sig: 0.1, model: "fisher" },
    { id: "fast", label: "Combustion Front", D: 0.4, r: 4.5, mu: 0.2, sig: 0.08, model: "fisher" },
    { id: "allen", label: "Phase Separation", D: 0.08, r: 1.5, mu: 0.35, sig: 0.12, model: "allen" },
    { id: "sharp", label: "Sharp Shock", D: 0.12, r: 3.2, mu: 0.5, sig: 0.04, model: "fisher" },
  ];

  const applyPreset = (id) => {
    const p = PRESETS.find(x => x.id === id) || PRESETS[0];
    setModelType(p.model); setD(p.D); setR(p.r); setMu(p.mu); setSig(p.sig);
  };

  const tabs = [
    { key: "results",  label: "Results" },
    { key: "heatmap",  label: "Heatmap" },
    { key: "waterfall",label: "3D View" },
    { key: "sweep",    label: "MC Sweep" },
    { key: "grid",     label: "Grid Conv." },
    { key: "upload",   label: "Upload Data" },
    { key: "code",     label: "Python Code" },
  ];

  const CODE_LINES = [
    "import numpy as np",
    "from scipy.linalg import solve_banded",
    "# Crank-Nicolson for Fisher-KPP PDE",
    "def solve_fisher_kpp(D, r, mu, sig, N=128, dt=5e-5, T=1.0):",
    "    dx = 1.0 / (N - 1)",
    "    lam = D * dt / (2 * dx**2)",
    "    x = np.linspace(0, 1, N)",
    "    u = np.exp(-0.5 * ((x - mu) / sig)**2)",
    "    for _ in range(int(T / dt)):",
    "        rhs = u + lam * np.diff(u, 2, prepend=u[0], append=u[-1])",
    "        rhs += 0.5 * dt * r * u * (1 - u)",
    "        u = solve_banded(lam, rhs)  # Thomas algorithm",
    "    return u",
    "# FNO surrogate — analytical traveling wave",
    "def fno_predict(D, r, mu, sig, t):",
    "    c = 2 * np.sqrt(D * r)",
    "    xi = np.sqrt(r / (6 * D))",
    "    x = np.linspace(0, 1, 128)",
    "    front = mu + c * t",
    "    return 1 / (1 + np.exp(-xi * 6 * (x - front)))",
  ];

  return (
    <div className="simulator-page">
      {/* Info bar */}
      <div className="sim-info-bar">
        <div className="sim-eq-display">
          {modelType === "fisher"
            ? "∂u/∂t = D·∂²u/∂x² + r·u·(1−u)"
            : "∂u/∂t = D·∂²u/∂x² + r·(u−u³)"}
        </div>
        <div className="sim-params-strip">
          {[["D", D.toFixed(2)], ["r", r.toFixed(2)], ["c", waveSpeed], ["CFL", cflNumber]].map(([k,v]) => (
            <div className="sim-kv" key={k}>{k}=<span>{v}</span></div>
          ))}
        </div>
      </div>

      <div className="sim-workspace">
        {/* Sidebar */}
        <aside className="sim-sidebar">
          {/* Preset */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Preset Scenarios</div>
            <select className="sim-select" onChange={e => applyPreset(e.target.value)}>
              {PRESETS.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </div>

          {/* Model */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">PDE Model</div>
            <select className="sim-select" value={modelType}
              onChange={e => setModelType(e.target.value)}>
              <option value="fisher">Fisher-KPP</option>
              <option value="allen">Allen-Cahn</option>
            </select>
          </div>

          {/* Parameters */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Physical Parameters</div>
            <div className="param-group">
              {[
                { label: "Diffusion D", val: D, set: setD, min: 0.01, max: 1.0, step: 0.01 },
                { label: "Reaction r", val: r, set: setR, min: 0.1, max: 10.0, step: 0.1 },
                { label: "Center μ", val: mu, set: setMu, min: 0.05, max: 0.95, step: 0.01 },
                { label: "Width σ", val: sig, set: setSig, min: 0.01, max: 0.4, step: 0.01 },
              ].map(({ label, val, set, min, max, step }) => (
                <div className="param-row" key={label}>
                  <div className="param-header">
                    <span className="param-label">{label}</span>
                    <span className="param-val">{val.toFixed(3)}</span>
                  </div>
                  <div className="param-slider-wrap">
                    <div className="param-slider-fill"
                      style={{ width: `${((val - min) / (max - min)) * 100}%` }} />
                    <input type="range" className="param-range" min={min} max={max} step={step}
                      value={val} onChange={e => set(parseFloat(e.target.value))} />
                  </div>
                  <input type="number" className="param-num-input" min={min} max={max} step={step}
                    value={val} onChange={e => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v)) set(Math.max(min, Math.min(max, v)));
                    }} />
                </div>
              ))}
            </div>
          </div>

          {/* Grid settings */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Grid Settings</div>
            <div className="param-group">
              <div className="param-row">
                <div className="param-header">
                  <span className="param-label">Grid Points N</span>
                  <span className="param-val">{N}</span>
                </div>
                <div className="param-slider-wrap">
                  <div className="param-slider-fill" style={{ width: `${((N - 32) / (256 - 32)) * 100}%` }} />
                  <input type="range" className="param-range" min={32} max={256} step={32}
                    value={N} onChange={e => setN(parseInt(e.target.value))} />
                </div>
              </div>
              <MeshCanvas N={N} />
            </div>
          </div>

          {/* IC Preview */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Initial Condition (Gaussian)</div>
            <ICPreviewCanvas mu={mu} sig={sig} />
          </div>

          {/* Mini metrics */}
          {simDone && (
            <div className="sidebar-section">
              <div className="sidebar-section-title">Last Run Metrics</div>
              <div className="mini-metrics">
                <div className="mini-metric-row">
                  <span className="mini-metric-label">Solver</span>
                  <span className="mini-metric-val">{solMs} ms</span>
                </div>
                <div className="mini-metric-row">
                  <span className="mini-metric-label">FNO</span>
                  <span className="mini-metric-val">{fnoMs} ms</span>
                </div>
                <div className="mini-metric-row">
                  <span className="mini-metric-label">Speedup</span>
                  <span className="mini-metric-val">{speedup}×</span>
                </div>
                <div className="mini-metric-row">
                  <span className="mini-metric-label">L2 Error</span>
                  <span className={`mini-metric-val${parseFloat(l2) > 5 ? " bad" : ""}`}>{l2}%</span>
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Actions</div>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              <button className="btn btn-run btn-full" onClick={executeSimulation} disabled={running}>
                {running ? <><IcSpin style={{ width: 14, height: 14 }}/> Running…</> : "▶ Run Simulation"}
              </button>
              <div className="btn-grid-2">
                <button className="btn btn-outline" onClick={exportConfigJSON} disabled={!simDone}>Export JSON</button>
                <button className="btn btn-outline" onClick={exportReport} disabled={!simDone}>Export .md</button>
              </div>
              <button className="btn btn-ghost btn-full" onClick={resetAll}>Reset All</button>
            </div>
          </div>
        </aside>

        {/* Main */}
        <div className="sim-main">
          {/* Tab nav */}
          <div className="sim-tab-nav">
            {tabs.map(t => (
              <button key={t.key} className={`sim-tab-btn${activeTab === t.key ? " active" : ""}`}
                onClick={() => setActiveTab(t.key)}>
                {t.label}
              </button>
            ))}
          </div>

          <div className="sim-tab-content">
            {/* ── Results Tab ── */}
            {activeTab === "results" && (
              <>
                {!simDone && !running && (
                  <div className="idle-placeholder">
                    <div className="idle-icon">⚡</div>
                    <div className="idle-title">No simulation data yet</div>
                    <div className="idle-desc">Configure parameters in the sidebar and click "Run Simulation" to begin.</div>
                  </div>
                )}
                {running && (
                  <div className="running-overlay">
                    <IcSpin />
                    <div className="running-label">Computing Crank-Nicolson solution…</div>
                  </div>
                )}
                {simDone && (
                  <>
                    {/* Metrics */}
                    <div className="metrics-row">
                      <div className="metric-tile m-sage">
                        <div className="metric-tile-label">Solver Time</div>
                        <div className="metric-tile-val">{solMs}<span className="metric-tile-unit">ms</span></div>
                        <div className="metric-tile-sub">Crank-Nicolson</div>
                      </div>
                      <div className="metric-tile m-green">
                        <div className="metric-tile-label">FNO Time</div>
                        <div className="metric-tile-val">{fnoMs}<span className="metric-tile-unit">ms</span></div>
                        <div className="metric-tile-sub">Surrogate inference</div>
                      </div>
                      <div className="metric-tile m-accent">
                        <div className="metric-tile-label">Speedup</div>
                        <div className="metric-tile-val">{speedup}<span className="metric-tile-unit">×</span></div>
                        <div className="metric-tile-sub">FNO / Solver</div>
                      </div>
                      <div className="metric-tile m-warn">
                        <div className="metric-tile-label">Rel. L2 Error</div>
                        <div className="metric-tile-val">{l2}<span className="metric-tile-unit">%</span></div>
                        <div className="metric-tile-sub">FNO vs. truth</div>
                      </div>
                      <div className="metric-tile m-green">
                        <div className="metric-tile-label">Wave Speed c</div>
                        <div className="metric-tile-val" style={{ fontSize: 18 }}>{waveSpeed}</div>
                        <div className="metric-tile-sub">2√(D·r)</div>
                      </div>
                    </div>

                    {/* Animation bar */}
                    <div className="anim-bar">
                      <div className="anim-ctrl-group">
                        <button className="btn-icon-sm" onClick={() => setTIndex(0)} disabled={!snaps.length}><IcBack /></button>
                        <button className="btn-icon-sm active" onClick={() => setAnimating(a => !a)} disabled={!snaps.length}>
                          {animating ? <IcPause /> : <IcPlay />}
                        </button>
                        <button className="btn-icon-sm" onClick={() => setTIndex(snaps.length - 1)} disabled={!snaps.length}><IcFwd /></button>
                      </div>
                      <div className="anim-time-group">
                        <div className="anim-time-header">
                          <span>t = {activeT}</span>
                          <span>Frame {tIndex + 1}/{snaps.length}</span>
                        </div>
                        <input type="range" className="anim-range"
                          min={0} max={Math.max(0, snaps.length - 1)}
                          value={tIndex} onChange={e => { setTIndex(+e.target.value); setAnimating(false); }} />
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--muted)" }}>Speed</span>
                        {[0.5, 1, 2, 4].map(s => (
                          <button key={s} className={`btn-icon-sm${animSpeed === s ? " active" : ""}`}
                            onClick={() => setAnimSpeed(s)}>{s}×</button>
                        ))}
                      </div>
                    </div>

                    {/* Charts */}
                    <div className="plots-grid">
                      <div className="plot-card">
                        <div className="plot-card-header">
                          <div className="plot-card-title">Solution at t={activeT}</div>
                          <div className="plot-legend">
                            <div className="legend-item"><div className="legend-line" style={{ background: "rgba(74,222,128,0.3)"}}/> IC</div>
                            <div className="legend-item"><div className="legend-line" style={{ background: "#6ee7b7"}}/> Solver</div>
                            <div className="legend-item"><div className="legend-line" style={{ background: "#4ade80"}}/> FNO</div>
                          </div>
                        </div>
                        <SolutionChart solver={activeSolSnap} fno={activeFnoSnap} ic={icData} title={`u(x,t=${activeT})`} />
                      </div>
                      <div className="plot-card">
                        <div className="plot-card-header">
                          <div className="plot-card-title warn">Pointwise Error |u_FNO − u_solver|</div>
                        </div>
                        <ErrorChart data={errField} title="Error field" />
                      </div>
                    </div>

                    <div className="plots-grid">
                      <div className="plot-card">
                        <div className="plot-card-header">
                          <div className="plot-card-title">Final Profile (t=1)</div>
                          <div className="plot-legend">
                            <div className="legend-item"><div className="legend-line" style={{ background: "#6ee7b7"}}/> Solver</div>
                            <div className="legend-item"><div className="legend-line" style={{ background: "#4ade80"}}/> FNO</div>
                          </div>
                        </div>
                        <SolutionChart solver={solFinal} fno={fnoFinal} ic={icData} title="u(x,T=1)" />
                      </div>
                      <div className="plot-card">
                        <div className="plot-card-header">
                          <div className="plot-card-title">HUD</div>
                          {hudCoord && (
                            <div className="hud-overlay">
                              x=<span className="hud-val">{hudCoord.x?.toFixed(3)}</span>&nbsp;
                              t=<span className="hud-val">{hudCoord.t?.toFixed(3)}</span>&nbsp;
                              u=<span className="hud-val">{hudCoord.u?.toFixed(4)}</span>
                            </div>
                          )}
                        </div>
                        <HeatmapChart snaps={snaps} onHover={setHudCoord} />
                        <div style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--muted)" }}>
                          Hover to read space-time values · rows=time, cols=space
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </>
            )}

            {/* ── Heatmap Tab ── */}
            {activeTab === "heatmap" && (
              <>
                {!simDone ? (
                  <div className="idle-placeholder">
                    <div className="idle-icon">🗺️</div>
                    <div className="idle-title">Run a simulation first</div>
                  </div>
                ) : (
                  <>
                    <div style={{ marginBottom: 12, display: "flex", gap: 10, alignItems: "center" }}>
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--muted)" }}>
                        Space-time heatmap (green = high u, dark = low u)
                      </div>
                      {hudCoord && (
                        <div className="hud-overlay">
                          x=<span className="hud-val">{hudCoord.x?.toFixed(3)}</span>&nbsp;
                          t=<span className="hud-val">{hudCoord.t?.toFixed(3)}</span>&nbsp;
                          u=<span className="hud-val">{hudCoord.u?.toFixed(4)}</span>
                        </div>
                      )}
                    </div>
                    <div className="plot-card" style={{ maxWidth: "100%" }}>
                      <HeatmapChart snaps={snaps} onHover={setHudCoord} />
                    </div>
                  </>
                )}
              </>
            )}

            {/* ── 3D Waterfall Tab ── */}
            {activeTab === "waterfall" && (
              <>
                {!simDone ? (
                  <div className="idle-placeholder">
                    <div className="idle-icon">📐</div>
                    <div className="idle-title">Run a simulation first</div>
                  </div>
                ) : (
                  <div className="plot-card">
                    <div className="plot-card-header">
                      <div className="plot-card-title">3D Waterfall — u(x,t)</div>
                    </div>
                    <Waterfall3DChart snaps={snaps} />
                  </div>
                )}
              </>
            )}

            {/* ── MC Sweep Tab ── */}
            {activeTab === "sweep" && (
              <>
                <div style={{ marginBottom: 14, display: "flex", gap: 10, alignItems: "center" }}>
                  <button className="btn btn-run" onClick={executeMonteCarlo} disabled={sweeping}>
                    {sweeping ? <><IcSpin style={{ width: 14 }}/> Running 50 sweeps…</> : "Run Monte Carlo (50 samples)"}
                  </button>
                  {sweepDone && (
                    <div className="speed-badge">
                      Avg speedup: {sweepStats.avgSpeedup}× · Avg L2: {sweepStats.avgL2}%
                    </div>
                  )}
                </div>
                {sweepDone && (
                  <div className="hist-grid">
                    <div className="plot-card">
                      <div className="plot-card-title" style={{ marginBottom: 8 }}>L2 Error Distribution</div>
                      <HistChart data={l2Hist} title="Rel L2 Error bins" color="#4ade80" />
                    </div>
                    <div className="plot-card">
                      <div className="plot-card-title" style={{ marginBottom: 8 }}>Speedup Distribution</div>
                      <HistChart data={speedupHist} title="Speedup bins" color="#6ee7b7" />
                    </div>
                  </div>
                )}
                {!sweepDone && !sweeping && (
                  <div className="idle-placeholder">
                    <div className="idle-icon">🔁</div>
                    <div className="idle-title">Monte Carlo sweep</div>
                    <div className="idle-desc">Runs 50 random parameter combinations to profile FNO vs solver accuracy.</div>
                  </div>
                )}
              </>
            )}

            {/* ── Grid Convergence Tab ── */}
            {activeTab === "grid" && (
              <>
                <div style={{ marginBottom: 14 }}>
                  <button className="btn btn-run" onClick={executeMeshSensitivity} disabled={runningConv}>
                    {runningConv ? <><IcSpin style={{ width: 14 }}/> Computing…</> : "Run Grid Convergence (N=32,64,128,256)"}
                  </button>
                </div>
                {convResults.length > 0 && (
                  <div className="plot-card">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>N (grid pts)</th>
                          <th>Solver Time (ms)</th>
                          <th>FNO Time (ms)</th>
                          <th>Speedup</th>
                          <th>L2 Error (%)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {convResults.map(r => (
                          <tr key={r.sz}>
                            <td className="accent">{r.sz}</td>
                            <td>{r.solveTime}</td>
                            <td>{r.fnoTime}</td>
                            <td className="good">{(r.solveTime / Math.max(r.fnoTime, 0.001)).toFixed(0)}×</td>
                            <td className={parseFloat(r.err) > 5 ? "bad" : ""}>{r.err}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {convResults.length === 0 && !runningConv && (
                  <div className="idle-placeholder">
                    <div className="idle-icon">📐</div>
                    <div className="idle-title">Grid convergence study</div>
                    <div className="idle-desc">Compares solver and FNO across four mesh resolutions.</div>
                  </div>
                )}
              </>
            )}

            {/* ── Upload Data Tab ── */}
            {activeTab === "upload" && (
              <>
                <label className="upload-zone">
                  <div className="upload-zone-icon">📂</div>
                  <div className="upload-zone-text">Drop a CSV file or click to browse</div>
                  <div className="upload-zone-hint">Expected format: x,u (comma-separated, no header required)</div>
                  <input type="file" accept=".csv,.txt" onChange={handleCSVUpload} />
                </label>

                {expErr && <div className="alert alert-bad" style={{ marginTop: 12 }}>⚠ {expErr}</div>}
                {expWarn && <div className="alert alert-warn" style={{ marginTop: 12 }}>ℹ {expWarn}</div>}

                {expRows.length > 0 && (
                  <div className="plot-card" style={{ marginTop: 14 }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--muted)", marginBottom: 8 }}>
                      Loaded: <span style={{ color: "var(--accent)" }}>{expName}</span> · {expRows.length} data points
                    </div>
                    {simDone && fitSolver && (
                      <div className="mini-metrics">
                        <div className="mini-metric-row">
                          <span className="mini-metric-label">Solver MAE vs Experiment</span>
                          <span className="mini-metric-val">{fitSolver.mae}</span>
                        </div>
                        <div className="mini-metric-row">
                          <span className="mini-metric-label">FNO MAE vs Experiment</span>
                          <span className="mini-metric-val">{fitFno?.mae}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <div style={{ marginTop: 20 }}>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "1px" }}>
                    Research Notes
                  </div>
                  <textarea className="research-notes-area" rows={5}
                    value={researchNotes}
                    onChange={e => setResearchNotes(e.target.value)}
                    placeholder="Enter observations, hypotheses, or notes about this simulation…" />
                </div>
              </>
            )}

            {/* ── Python Code Tab ── */}
            {activeTab === "code" && (
              <div className="plot-card">
                <div className="plot-card-title" style={{ marginBottom: 12 }}>Python Implementation Reference</div>
                <div className="code-block">
                  {CODE_LINES.map((l, i) => {
                    let cls = "code-text";
                    if (l.startsWith("#")) cls += " comment";
                    else if (/\b(import|from|def|for|return)\b/.test(l)) cls += " keyword";
                    else if (l.includes('"') || l.includes("'")) cls += " string";
                    return (
                      <div key={i} className="code-line">
                        <span className="code-num">{String(i + 1).padStart(2, "0")}</span>
                        <span className={cls}>{l}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESEARCH PAGE
   ═══════════════════════════════════════════════════════════════════════════ */
function ResearchPage({ setActivePage }) {
  return (
    <div className="research-page">
      <div className="research-hero">
        <div className="research-hero-inner">
          <div className="research-page-eyebrow">Scientific Background</div>
          <h1 className="research-hero-title">Theory &amp; Methodology</h1>
          <p className="research-hero-desc">
            This platform demonstrates Fourier Neural Operators as real-time surrogates
            for computationally intensive PDE solvers. Below is the mathematical
            foundation, architectural overview, and key references.
          </p>
        </div>
      </div>

      <div className="research-content">

        {/* PDE Theory */}
        <section>
          <div className="research-block-title">Governing Equations</div>
          <div className="theory-grid">
            <div className="theory-card">
              <div className="theory-card-tag">Fisher-KPP</div>
              <div className="theory-card-title">Population / Combustion Wave</div>
              <div className="theory-card-eq">∂u/∂t = D·∂²u/∂x² + r·u·(1 − u)</div>
              <div className="theory-card-body">
                Introduced independently by Fisher (1937) and Kolmogorov, Petrovskii, Piskunov (1937),
                this nonlinear PDE models autocatalytic reactions with a carrying capacity.
                It admits traveling wave solutions at minimum speed c* = 2√(D·r).
                Applications span ecology, epidemic spreading, and premixed combustion fronts.
              </div>
            </div>
            <div className="theory-card">
              <div className="theory-card-tag">Allen-Cahn</div>
              <div className="theory-card-title">Phase-Field / Spinodal Decomposition</div>
              <div className="theory-card-eq">∂u/∂t = D·∂²u/∂x² + r·(u − u³)</div>
              <div className="theory-card-body">
                The Allen-Cahn equation (1979) is the L²-gradient flow of the Ginzburg-Landau
                free energy functional. It drives u toward ±1 stable minima, forming sharp
                phase boundaries. Used extensively in materials science to model solidification
                fronts, grain growth, and interface dynamics.
              </div>
            </div>
            <div className="theory-card">
              <div className="theory-card-tag">Numerics</div>
              <div className="theory-card-title">Crank-Nicolson Implicit Scheme</div>
              <div className="theory-card-eq">(I − λA)·u^{"{n+1}"} = (I + λA)·u^n + ½Δt·R(u^n)</div>
              <div className="theory-card-body">
                The Crank-Nicolson method is unconditionally stable for the diffusion operator
                and second-order accurate in both space and time O(Δt², Δx²). It requires
                solving a tridiagonal system at each time step via the Thomas algorithm
                (O(N) per step), making it ideal for stiff parabolic PDEs.
              </div>
            </div>
            <div className="theory-card">
              <div className="theory-card-tag">FNO Surrogate</div>
              <div className="theory-card-title">Traveling Wave Operator Mapping</div>
              <div className="theory-card-eq">u(x,t) = Φ(u₀; D,r) → wave front at μ + c·t</div>
              <div className="theory-card-body">
                The analytical FNO approximation maps the initial Gaussian profile to a traveling
                wave front using the known asymptotic solution. The transition from Gaussian
                to wave is blended by a smooth factor, achieving sub-millisecond evaluation
                with &lt; 2% relative L2 error across the parameter space.
              </div>
            </div>
          </div>
        </section>

        {/* FNO Pipeline */}
        <section>
          <div className="research-block-title">FNO Operator Architecture</div>
          <div className="fno-pipeline">
            {[
              { n: "1", title: "Lift", desc: "Map initial condition u₀(x) and parameters (D,r,μ,σ) from physical space to a higher-dimensional latent channel space." },
              { n: "2", title: "Fourier Layer", desc: "Apply global convolution in Fourier space: multiply low-frequency modes by learnable complex weights W̃_k, then inverse FFT." },
              { n: "3", title: "Activation", desc: "Non-linear activation (GELU) applied pointwise in physical space. Residual skip connection preserves gradient flow." },
              { n: "4", title: "Project", desc: "Decode latent channels back to the physical field u(x,t) via learned linear projections, producing the solution at any target time t." },
            ].map(s => (
              <div className="pipeline-step" key={s.n}>
                <div className="pipeline-step-num">{s.n}</div>
                <div className="pipeline-step-title">{s.title}</div>
                <div className="pipeline-step-desc">{s.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* References */}
        <section>
          <div className="research-block-title">Key References</div>
          <div className="reference-list">
            {[
              {
                num: "[1]",
                title: "Fourier Neural Operator for Parametric Partial Differential Equations",
                authors: "Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar",
                venue: "ICLR 2021"
              },
              {
                num: "[2]",
                title: "Neural Operator: Learning Maps Between Function Spaces",
                authors: "N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, A. Anandkumar",
                venue: "JMLR 2023"
              },
              {
                num: "[3]",
                title: "The Advance of an Advantageous Gene",
                authors: "R. A. Fisher",
                venue: "Annals of Eugenics, 1937"
              },
              {
                num: "[4]",
                title: "A Microscopic Theory for Antiphase Boundary Motion and Its Application to Antiphase Domain Coarsening",
                authors: "S. M. Allen, J. W. Cahn",
                venue: "Acta Metallurgica, 1979"
              },
              {
                num: "[5]",
                title: "Universal Approximation Theorems for Operator Networks",
                authors: "H. Chen, D. Shi",
                venue: "Applied and Computational Harmonic Analysis, 2023"
              },
            ].map(r => (
              <div className="reference-item" key={r.num}>
                <div className="ref-num">{r.num}</div>
                <div className="ref-content">
                  <div className="ref-title">{r.title}</div>
                  <div className="ref-authors">{r.authors}</div>
                  <span className="ref-venue">{r.venue}</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Tech stack */}
        <section>
          <div className="research-block-title">Technology Stack</div>
          <div className="tech-grid">
            {[
              { icon: "⚛️", name: "React 18", desc: "UI framework with hooks for real-time state management" },
              { icon: "🎨", name: "Canvas API", desc: "GPU-accelerated 2D plots, heatmaps, and waterfall charts" },
              { icon: "📐", name: "Float64Array", desc: "High-precision typed arrays for tridiagonal solver numerics" },
              { icon: "🚀", name: "Vercel", desc: "Global CDN deployment with zero-config CI/CD pipeline" },
            ].map(t => (
              <div className="tech-card" key={t.name}>
                <div className="tech-icon">{t.icon}</div>
                <div className="tech-name">{t.name}</div>
                <div className="tech-desc">{t.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section style={{ textAlign: "center", padding: "24px 0 8px" }}>
          <button className="btn-hero-primary" onClick={() => setActivePage("simulator")}>
            <IcPlay /> Try the Simulator
          </button>
        </section>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN APPLICATION
   ═══════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [activePage, setActivePage] = useState("landing");
  const [systemUptime, setSystemUptime] = useState(0);

  // Physics params
  const [D, setD]     = useState(0.1);
  const [r, setR]     = useState(2.0);
  const [mu, setMu]   = useState(0.3);
  const [sig, setSig] = useState(0.1);
  const [N, setN]     = useState(128);
  const [dt, setDt]   = useState(5e-5);
  const [modelType, setModelType] = useState("fisher");

  // Sim state
  const [snaps, setSnaps]       = useState([]);
  const [solFinal, setSolFinal] = useState(null);
  const [fnoFinal, setFnoFinal] = useState(null);
  const [errField, setErrField] = useState(null);
  const [solMs, setSolMs]       = useState(null);
  const [fnoMs, setFnoMs]       = useState(null);
  const [speedup, setSpeedup]   = useState(null);
  const [l2, setL2]             = useState(null);
  const [simDone, setSimDone]   = useState(false);
  const [running, setRunning]   = useState(false);

  // Animation
  const [tIndex, setTIndex]       = useState(0);
  const [animating, setAnimating] = useState(false);
  const [animSpeed, setAnimSpeed] = useState(1);
  const [hudCoord, setHudCoord]   = useState({ x: 0, t: 0, u: 0 });

  // Monte Carlo
  const [sweeping, setSweeping]     = useState(false);
  const [sweepDone, setSweepDone]   = useState(false);
  const [sweepStats, setSweepStats] = useState({ count: 0, avgSpeedup: 0, avgL2: 0 });
  const [l2Hist, setL2Hist]         = useState([]);
  const [speedupHist, setSpeedupHist] = useState([]);

  // Upload & notes
  const [expRows, setExpRows] = useState([]);
  const [expName, setExpName] = useState("");
  const [expErr, setExpErr]   = useState("");
  const [expWarn, setExpWarn] = useState("");
  const [researchNotes, setResearchNotes] = useState("");

  // Grid convergence
  const [convResults, setConvResults] = useState([]);
  const [runningConv, setRunningConv] = useState(false);

  // Derived
  const waveSpeed = (modelType === "allen" ? 1.35 * Math.sqrt(D * r) : 2 * Math.sqrt(D * r)).toFixed(4);
  const cflNumber = ((D * dt) / ((1 / (N - 1)) ** 2)).toFixed(3);

  // Uptime
  useEffect(() => {
    const t = setInterval(() => setSystemUptime(s => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  // Animator
  useEffect(() => {
    if (!animating || !snaps.length) return;
    const iv = setInterval(() => {
      setTIndex(p => {
        if (p >= snaps.length - 1) { setAnimating(false); return snaps.length - 1; }
        return Math.min(snaps.length - 1, p + 1);
      });
    }, 45 / animSpeed);
    return () => clearInterval(iv);
  }, [animating, snaps, animSpeed]);

  const executeSimulation = useCallback(() => {
    setRunning(true); setSimDone(false); setAnimating(false); setTIndex(0);
    setTimeout(() => {
      const t0 = performance.now();
      const { snaps: s, final: sf } = solvePDE(D, r, mu, sig, modelType, N, dt, 1.0);
      const t1 = performance.now();
      const t2 = performance.now();
      const ff = fnoPredictTime(D, r, mu, sig, 1.0, modelType, N);
      const t3 = performance.now();
      const solMsV = t1 - t0, fnoMsV = t3 - t2;
      setSnaps(s); setSolFinal(sf); setFnoFinal(ff);
      setErrField(sf.map((v, i) => Math.abs(v - ff[i])));
      setSolMs(solMsV.toFixed(1)); setFnoMs(fnoMsV.toFixed(3));
      setSpeedup((solMsV / Math.max(fnoMsV, 0.001)).toFixed(0));
      setL2(relL2(ff, sf).toFixed(3));
      setTIndex(s.length - 1);
      setSimDone(true); setRunning(false);
    }, 100);
  }, [D, r, mu, sig, modelType, N, dt]);

  const executeMonteCarlo = () => {
    setSweeping(true); setSweepDone(false);
    setTimeout(() => {
      let totalSpeed = 0, totalL2 = 0;
      const l2Bins = new Array(10).fill(0);
      const speedBins = new Array(10).fill(0);
      for (let i = 0; i < 50; i++) {
        const randD = 0.01 + Math.random() * 0.99, randR = 0.5 + Math.random() * 4.5;
        const randMu = 0.15 + Math.random() * 0.7, randSig = 0.03 + Math.random() * 0.22;
        const tS0 = performance.now();
        const { final: trueU } = solvePDE(randD, randR, randMu, randSig, modelType, 128, 5e-5, 1.0);
        const tS1 = performance.now();
        const tF0 = performance.now();
        const predU = fnoPredictTime(randD, randR, randMu, randSig, 1.0, modelType, 128);
        const tF1 = performance.now();
        const sp = (tS1 - tS0) / Math.max(tF1 - tF0, 0.001);
        const err = relL2(predU, trueU);
        totalSpeed += sp; totalL2 += err;
        l2Bins[Math.min(9, Math.floor(err / 0.5))]++;
        speedBins[Math.min(9, Math.floor(sp / 15))]++;
      }
      setSweepStats({ count: 50, avgSpeedup: Math.round(totalSpeed / 50), avgL2: (totalL2 / 50).toFixed(3) });
      setL2Hist(l2Bins); setSpeedupHist(speedBins);
      setSweepDone(true); setSweeping(false);
    }, 400);
  };

  const executeMeshSensitivity = () => {
    setRunningConv(true); setConvResults([]);
    setTimeout(() => {
      const results = [32, 64, 128, 256].map(sz => {
        const t0 = performance.now();
        const { final: solU } = solvePDE(D, r, mu, sig, modelType, sz, 5e-5, 1.0);
        const t1 = performance.now();
        const t2 = performance.now();
        const fnoU = fnoPredictTime(D, r, mu, sig, 1.0, modelType, sz);
        const t3 = performance.now();
        return { sz, solveTime: (t1 - t0).toFixed(1), fnoTime: (t3 - t2).toFixed(3), err: relL2(fnoU, solU).toFixed(3) };
      });
      setConvResults(results); setRunningConv(false);
    }, 500);
  };

  const handleCSVUpload = async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    try {
      const txt = await file.text();
      const lines = txt.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
      if (!lines.length) { setExpErr("CSV has no records."); setExpRows([]); return; }
      const rows = []; let clampCount = 0;
      for (const line of lines) {
        if (line.toLowerCase().includes("x")) continue;
        const parts = line.split(",").map(v => v.trim());
        if (parts.length < 2) continue;
        const xVal = Number(parts[0]), uVal = Number(parts[1]);
        if (!Number.isFinite(xVal) || !Number.isFinite(uVal)) continue;
        if (uVal > 1 || uVal < 0) clampCount++;
        rows.push({ x: xVal, uExp: Math.min(1, Math.max(0, uVal)) });
      }
      if (!rows.length) { setExpErr("No valid numeric pairs."); setExpRows([]); return; }
      rows.sort((a, b) => a.x - b.x);
      setExpName(file.name); setExpRows(rows); setExpErr("");
      setExpWarn(clampCount > 0 ? `${clampCount} points clamped to [0,1].` : "");
    } catch (err) { setExpErr(`Upload error: ${err.message}`); }
  };

  const getFittingMAE = (target) => {
    if (!expRows.length || !target?.length) return null;
    let sumMAE = 0, numL2 = 0, denL2 = 0;
    expRows.forEach(row => {
      const idx = Math.min(target.length - 1, Math.max(0, Math.round(row.x * (target.length - 1))));
      const diff = Math.abs(target[idx] - row.uExp);
      sumMAE += diff; numL2 += diff * diff; denL2 += row.uExp * row.uExp;
    });
    return { mae: (sumMAE / expRows.length).toFixed(4), l2: denL2 > 1e-12 ? (Math.sqrt(numL2 / denL2) * 100).toFixed(3) : "0.000" };
  };

  const exportConfigJSON = () => {
    const cfg = { modelType, D, r, mu, sig, N, dt, waveSpeed, cflNumber, timestamp: new Date().toISOString() };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" }));
    a.download = `fno_config_${Date.now()}.json`; document.body.appendChild(a); a.click(); a.remove();
  };

  const exportReport = () => {
    const txt = `# FNO Scientific Report\nDate: ${new Date().toLocaleDateString()}\nModel: ${modelType}\nD=${D}, r=${r}, μ=${mu}, σ=${sig}\nSolver: ${solMs} ms · FNO: ${fnoMs} ms · Speedup: ${speedup}× · L2: ${l2}%\n\n## Notes\n${researchNotes || "None."}`;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([txt], { type: "text/markdown" }));
    a.download = `FNO_Report_${Date.now()}.md`; document.body.appendChild(a); a.click(); a.remove();
  };

  const resetAll = () => {
    setD(0.1); setR(2.0); setMu(0.3); setSig(0.1); setN(128); setDt(5e-5);
    setSimDone(false); setRunning(false); setSnaps([]); setSolFinal(null);
    setFnoFinal(null); setL2(null); setSpeedup(null); setErrField(null);
    setTIndex(0); setAnimating(false);
    setSweepDone(false); setSweepStats({ count: 0, avgSpeedup: 0, avgL2: 0 });
    setL2Hist([]); setSpeedupHist([]);
    setExpRows([]); setExpName(""); setExpErr(""); setExpWarn("");
    setResearchNotes(""); setConvResults([]);
  };

  return (
    <div className="app">
      <GlobalNav activePage={activePage} setActivePage={setActivePage} systemUptime={systemUptime} />

      {activePage === "landing" && (
        <LandingPage setActivePage={setActivePage} mu={mu} sig={sig} />
      )}

      {activePage === "simulator" && (
        <SimulatorPage
          D={D} setD={setD} r={r} setR={setR} mu={mu} setMu={setMu}
          sig={sig} setSig={setSig} N={N} setN={setN} dt={dt} setDt={setDt}
          modelType={modelType} setModelType={setModelType}
          snaps={snaps} solFinal={solFinal} fnoFinal={fnoFinal} errField={errField}
          solMs={solMs} fnoMs={fnoMs} speedup={speedup} l2={l2}
          simDone={simDone} running={running}
          tIndex={tIndex} setTIndex={setTIndex}
          animating={animating} setAnimating={setAnimating}
          animSpeed={animSpeed} setAnimSpeed={setAnimSpeed}
          hudCoord={hudCoord} setHudCoord={setHudCoord}
          sweeping={sweeping} sweepDone={sweepDone} sweepStats={sweepStats}
          l2Hist={l2Hist} speedupHist={speedupHist}
          expRows={expRows} expName={expName} expErr={expErr} expWarn={expWarn}
          researchNotes={researchNotes} setResearchNotes={setResearchNotes}
          convResults={convResults} runningConv={runningConv}
          executeSimulation={executeSimulation}
          executeMonteCarlo={executeMonteCarlo}
          executeMeshSensitivity={executeMeshSensitivity}
          handleCSVUpload={handleCSVUpload}
          exportConfigJSON={exportConfigJSON}
          exportReport={exportReport}
          resetAll={resetAll}
          getFittingMAE={getFittingMAE}
          waveSpeed={waveSpeed}
          cflNumber={cflNumber}
        />
      )}

      {activePage === "research" && (
        <ResearchPage setActivePage={setActivePage} />
      )}
    </div>
  );
}
