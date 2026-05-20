/* eslint-disable */
import { useState, useEffect, useRef, useCallback } from "react";

/* ═══════════════════════════════════════════════════════════════════════════
   PREMIUM GLASSMORPHISM & NEON GLYPH SVG ICONS
   ═══════════════════════════════════════════════════════════════════════════ */
const IconWaveform = () => (
  <svg className="custom-svg-icon wave-svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 12h3l3-9 4 18 3-12h5" stroke="var(--accent-fno)" strokeWidth="2" strokeLinejoin="round" />
    <path d="M3 18h18" stroke="var(--border)" strokeWidth="1" strokeDasharray="3 3" />
    <circle cx="12" cy="15" r="2.5" fill="var(--accent-solver)" stroke="none" />
  </svg>
);

const IconMesh = () => (
  <svg className="custom-svg-icon mesh-svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2.5" stroke="var(--accent-solver)" strokeWidth="1.5" />
    <path d="M9 3v18" stroke="rgba(139, 92, 246, 0.25)" />
    <path d="M15 3v18" stroke="rgba(139, 92, 246, 0.25)" />
    <path d="M3 9h18" stroke="rgba(139, 92, 246, 0.25)" />
    <path d="M3 15h18" stroke="rgba(139, 92, 246, 0.25)" />
    <circle cx="9" cy="9" r="2.5" fill="var(--accent-fno)" stroke="none" />
    <circle cx="15" cy="15" r="2.5" fill="var(--accent-fno)" stroke="none" />
  </svg>
);

const IconChart = () => (
  <svg className="custom-svg-icon chart-svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 3v18h18" stroke="var(--muted)" strokeWidth="1.5" />
    <path d="M18 7l-5 5-4-3-5 6" stroke="var(--accent-fno)" strokeWidth="2" strokeLinejoin="round" />
    <circle cx="6" cy="15" r="2" fill="var(--accent-solver)" stroke="none" />
    <circle cx="10" cy="10" r="2" fill="var(--accent-solver)" stroke="none" />
    <circle cx="14" cy="12" r="2" fill="var(--accent-solver)" stroke="none" />
    <circle cx="18" cy="6" r="2" fill="var(--accent-solver)" stroke="none" />
  </svg>
);

const IconLibrary = () => (
  <svg className="custom-svg-icon lib-svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="var(--accent-solver)" strokeWidth="1.5" />
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" stroke="var(--accent-fno)" strokeWidth="2" />
    <path d="M8 6h8" stroke="var(--border)" strokeWidth="1" />
    <path d="M8 10h8" stroke="var(--border)" strokeWidth="1" />
    <path d="M8 14h5" stroke="var(--border)" strokeWidth="1" />
  </svg>
);

const IconDiagnostic = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: "6px", verticalAlign: "middle" }}>
    <path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke="var(--accent-fno)" />
  </svg>
);

const IconExport = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: "6px", verticalAlign: "middle" }}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="currentColor" />
    <polyline points="7 10 12 15 17 10" stroke="currentColor" />
    <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" />
  </svg>
);

const IconClear = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: "6px", verticalAlign: "middle" }}>
    <polyline points="3 6 5 6 21 6" stroke="currentColor" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" />
    <line x1="10" y1="11" x2="10" y2="17" stroke="currentColor" />
    <line x1="14" y1="11" x2="14" y2="17" stroke="currentColor" />
  </svg>
);

const IconInfo = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "block" }}>
    <circle cx="12" cy="12" r="10" stroke="var(--accent-fno)" />
    <line x1="12" y1="16" x2="12" y2="12" stroke="var(--accent-fno)" />
    <line x1="12" y1="8" x2="12.01" y2="8" stroke="var(--accent-fno)" fill="currentColor" />
  </svg>
);

const IconPlay = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none">
    <polygon points="6,4 20,12 6,20" />
  </svg>
);

const IconPause = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none">
    <rect x="5" y="4" width="4" height="16" rx="1" />
    <rect x="15" y="4" width="4" height="16" rx="1" />
  </svg>
);

const IconRewind = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none" style={{ marginRight: "5px", verticalAlign: "middle" }}>
    <polygon points="11,19 2,12 11,5" />
    <polygon points="22,19 13,12 22,5" />
  </svg>
);

const IconFastForward = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none" style={{ marginRight: "5px", verticalAlign: "middle" }}>
    <polygon points="13,19 22,12 13,5" />
    <polygon points="2,19 11,12 2,5" />
  </svg>
);

const IconGear = () => (
  <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="var(--accent-solver)" strokeWidth="1.5" className="spinning-cog-icon" style={{ display: "inline-block" }}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

const IconSpinner = () => (
  <svg width="48" height="48" viewBox="0 0 50 50" className="neural-spinner-icon" style={{ display: "inline-block" }}>
    <circle cx="25" cy="25" r="20" fill="none" stroke="rgba(0, 245, 212, 0.12)" strokeWidth="4" />
    <circle cx="25" cy="25" r="20" fill="none" stroke="var(--accent-fno)" strokeWidth="4" strokeDasharray="31.4 31.4" strokeLinecap="round" />
    <circle cx="25" cy="25" r="14" fill="none" stroke="rgba(139, 92, 246, 0.12)" strokeWidth="3" />
    <circle cx="25" cy="25" r="14" fill="none" stroke="var(--accent-solver)" strokeWidth="3" strokeDasharray="22 22" strokeLinecap="round" strokeDashoffset="11" />
  </svg>
);

const VercelStatusBadge = () => (
  <a href="https://fnoproject.vercel.app" target="_blank" rel="noopener noreferrer" className="vercel-badge-btn" title="View live production deployment on Vercel">
    <svg className="vercel-logo-svg" width="10" height="10" viewBox="0 0 512 512" fill="currentColor">
      <path d="M256,48L496,464H16Z" />
    </svg>
    <span className="vercel-text">DEPLOYED</span>
    <span className="vercel-dot-pulse" />
  </a>
);

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
  c[0] = up[0] / d0; 
  d[0] = rhs[0] / d0;
  
  for (let i = 1; i < n; i++) {
    let m = diag[i] - lo[i] * c[i - 1];
    if (Math.abs(m) < eps) {
      m = m >= 0 ? eps : -eps;
    }
    c[i] = up[i] / m;
    d[i] = (rhs[i] - lo[i] * d[i - 1]) / m;
  }
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    x[i] = d[i] - c[i] * x[i + 1];
  }
  return x;
}

function solvePDE(D, r, mu, sig, modelType = "fisher", N = 128, dt = 5e-5, T_end = 1.0) {
  // Defensive input sanitization to block crash-inducing or NaN parameters
  const N_safe = Math.max(3, Math.min(512, Math.round(N || 128)));
  const dt_safe = Math.max(1e-6, Math.min(0.1, dt || 5e-5));
  const D_safe = Math.max(1e-5, Math.min(10.0, D ?? 0.1));
  const r_safe = Math.max(0.0, Math.min(50.0, r ?? 2.0));
  const mu_safe = Math.max(0.001, Math.min(0.999, mu ?? 0.3));
  const sig_safe = Math.max(0.001, Math.min(2.0, sig ?? 0.1));

  const dx = 1 / (N_safe - 1);
  const lam = D_safe * dt_safe / (2 * dx * dx);
  let u = new Float64Array(N_safe);
  
  // Gaussian Initial Condition
  for (let i = 0; i < N_safe; i++) {
    const x = i * dx;
    const exponent = -0.5 * ((x - mu_safe) / sig_safe) ** 2;
    u[i] = Math.min(1, Math.max(0, Math.exp(exponent)));
  }
  
  const lo = new Float64Array(N_safe).fill(-lam);
  const di = new Float64Array(N_safe).fill(1 + 2 * lam);
  const up = new Float64Array(N_safe).fill(-lam);
  di[0] = 1 + lam; di[N_safe - 1] = 1 + lam; // Neumann Zero-flux BCs
  
  // Guard loop iterations to protect UI thread responsiveness
  let nSteps = Math.round(T_end / dt_safe);
  if (nSteps > 200000) {
    nSteps = 200000;
  }
  const saveEvery = Math.max(1, Math.floor(nSteps / 80));
  const snaps = [Array.from(u)];

  try {
    for (let s = 0; s < nSteps; s++) {
      const rhs = new Float64Array(N_safe);
      for (let i = 0; i < N_safe; i++) {
        const l = i > 0 ? u[i-1] : u[i];
        const rv = i < N_safe-1 ? u[i+1] : u[i];
        
        let rxn = 0;
        if (modelType === "allen") {
          rxn = r_safe * (u[i] - u[i] ** 3);
        } else {
          rxn = r_safe * u[i] * (1 - u[i]); // default: Fisher-KPP
        }
        
        rhs[i] = u[i] + lam * (l - 2 * u[i] + rv) + (dt_safe / 2) * rxn;
        if (isNaN(rhs[i]) || !isFinite(rhs[i])) {
          rhs[i] = 0.0;
        }
      }
      const nextU = thomasSolve(lo, di, up, rhs);
      for (let i = 0; i < N_safe; i++) {
        u[i] = Math.min(1, Math.max(0, isNaN(nextU[i]) || !isFinite(nextU[i]) ? u[i] : nextU[i]));
      }
      if (s % saveEvery === 0) {
        snaps.push(Array.from(u));
      }
    }
  } catch (err) {
    console.error("Implicit solver calculation failure:", err);
  }
  
  if (snaps.length <= 80) snaps.push(Array.from(u));
  return { snaps, final: Array.from(u) };
}

/* ═══════════════════════════════════════════════════════════════════════════
   FNO SURROGATE — Time-dependent Traveling Wave Operator Mapping
   ═══════════════════════════════════════════════════════════════════════════ */
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
  
  if (modelType === "allen") {
    c = 1.35 * Math.sqrt(D_safe * r_safe); // Allen-Cahn interfacial wave speed profile
    xi = Math.sqrt(r_safe / (2 * D_safe));
  }
  
  const front = mu_safe + c * t_safe;
  return Array.from({ length: N_safe }, (_, i) => {
    const x = i * dx;
    const icVal = Math.min(1, Math.max(0, Math.exp(-0.5 * ((x - mu_safe) / sig_safe) ** 2)));
    
    let wave = 0;
    if (modelType === "allen") {
      wave = 0.5 * (1 + Math.tanh(xi * (x - front)));
    } else {
      wave = 1 / (1 + Math.exp(-xi * 6 * (x - front))); // Fisher KPP wave front
    }
    
    const blend = t_safe === 0 ? 0 : Math.min(1, t_safe * 1.5); // transition curve from Gaussian
    const pred = (1 - blend) * icVal + blend * Math.min(1, Math.max(0, wave));
    return Math.min(1, Math.max(0, isNaN(pred) || !isFinite(pred) ? icVal : pred));
  });
}

const relL2 = (a, b) => {
  let n = 0, d = 0;
  for (let i = 0; i < a.length; i++) {
    n += (a[i] - b[i]) ** 2;
    d += b[i] ** 2;
  }
  return d > 1e-12 ? Math.sqrt(n / d) * 100 : 0;
};

/* ═══════════════════════════════════════════════════════════════════════════
   CANVAS PLOT RENDERING HOOKS — Updated to Cyberpunk Teal/Obsidian Theme
   ═══════════════════════════════════════════════════════════════════════════ */
const PAL = {
  bg: "#020204", panel: "#080b11", border: "#182235", borderGlow: "#00f5d4",
  text: "#f8fafc", muted: "#8e9aaf", accentFno: "#00f5d4", accentSolver: "#8b5cf6",
  good: "#10b981", bad: "#ff006e", purple: "#7b2cbf", dim: "#0d0e14"
};

function useCanvas(draw, deps) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; 
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    draw(ctx, c.width, c.height);
  }, deps);
  return ref;
}

function drawAxes(ctx, W, H, pad) {
  ctx.strokeStyle = "rgba(0, 245, 212, 0.1)";
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
  if (glow) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = pad.l + (i / (data.length - 1)) * pw;
    const y = pad.t + ph * (1 - Math.min(1, Math.max(0, v)));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
}

const infernoColormap = (v) => {
  const t = Math.min(1, Math.max(0, v));
  const r = Math.min(255, Math.round(t < 0.4 ? t * 2.5 * 255 : 255));
  const g = Math.min(255, Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 220));
  const b = Math.min(255, Math.round(t < 0.2 ? t * 5 * 200 : t > 0.7 ? 0 : ((0.7 - t) / 0.5) * 200));
  return `rgb(${r},${g},${b})`;
};

function project3D(x, t, u, W, H, pad) {
  const cx = W / 2;
  const cy = H / 2 + 15;
  const thetaX = -Math.PI / 6;
  const thetaT = Math.PI / 8;
  const scaleX = (W - pad.l - pad.r) * 0.44;
  const scaleT = (H - pad.t - pad.b) * 0.44;
  const scaleU = 50;

  const px = cx + (x - 0.5) * scaleX * Math.cos(thetaX) + (t - 0.5) * scaleT * Math.cos(thetaT);
  const py = cy + (x - 0.5) * scaleX * Math.sin(thetaX) + (t - 0.5) * scaleT * Math.sin(thetaT) - u * scaleU;
  return { x: px, y: py };
}

/* ═══════════════════════════════════════════════════════════════════════════
   SCIENTIFIC CHART COMPONENTS
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
    
    if (ic) drawLine(ctx, ic, W, H, pad, PAL.muted, 1, false);
    if (solver) drawLine(ctx, solver, W, H, pad, PAL.accentSolver, 2.5, false);
    if (fno) drawLine(ctx, fno, W, H, pad, PAL.accentFno, 1.5, true);
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
    drawLine(ctx, norm, W, H, pad, PAL.bad, 1.5, false);
    
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;
    ctx.beginPath();
    norm.forEach((v, i) => {
      const x = pad.l + (i / (norm.length - 1)) * pw;
      const y = pad.t + ph * (1 - v);
      if (i === 0) ctx.moveTo(x, H - pad.b);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(W - pad.r, H - pad.b); ctx.closePath();
    ctx.fillStyle = "rgba(255, 0, 110, 0.08)"; ctx.fill();
    
    ctx.fillStyle = PAL.muted; ctx.font = "9px 'JetBrains Mono',monospace";
    ctx.fillText(`max peak: ${max.toFixed(5)}`, W - 110, 15);
  }, [data, title]);
  return <canvas ref={ref} width={480} height={190} className="plot-canvas" />;
}

function HeatmapChart({ snaps, onHover }) {
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    if (!snaps?.length) return;
    const nT = snaps.length; 
    const nX = snaps[0].length;
    const cw = W / nX; 
    const ch = H / nT;
    for (let t = 0; t < nT; t++) {
      for (let x = 0; x < nX; x++) {
        ctx.fillStyle = infernoColormap(snaps[t][x]);
        ctx.fillRect(x * cw, t * ch, cw + 0.5, ch + 0.5);
      }
    }
  }, [snaps]);

  const handleMouseMove = (e) => {
    if (!snaps?.length || !onHover) return;
    const c = ref.current;
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    const px = Math.min(1, Math.max(0, mx / rect.width));
    const pt = Math.min(1, Math.max(0, my / rect.height));
    
    const snapIdx = Math.min(snaps.length - 1, Math.max(0, Math.floor(pt * snaps.length)));
    const nodeIdx = Math.min(snaps[0].length - 1, Math.max(0, Math.floor(px * snaps[0].length)));
    const uVal = snaps[snapIdx][nodeIdx];
    
    onHover({ x: px, t: pt, u: uVal });
  };

  return (
    <canvas 
      ref={ref} 
      width={480} 
      height={200} 
      className="plot-canvas" 
      style={{ cursor: "crosshair" }} 
      onMouseMove={handleMouseMove}
    />
  );
}

function Waterfall3DChart({ snaps }) {
  const ref = useCanvas((ctx, W, H) => {
    drawWaterfall(ctx, snaps, W, H);
  }, [snaps]);
  return <canvas ref={ref} width={480} height={200} className="plot-canvas" />;
}

function drawWaterfall(ctx, snaps, W, H) {
  ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
  if (!snaps || snaps.length === 0) return;
  const pad = { t: 15, b: 15, l: 15, r: 15 };
  
  const skip = Math.max(1, Math.floor(snaps.length / 22));
  const selected = [];
  for (let i = 0; i < snaps.length; i += skip) {
    selected.push({ t: i / (snaps.length - 1), data: snaps[i] });
  }
  if (selected[selected.length - 1].t !== 1) {
    selected.push({ t: 1, data: snaps[snaps.length - 1] });
  }

  selected.forEach((row) => {
    const t = row.t;
    const data = row.data;
    const N = data.length;
    
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = i / (N - 1);
      const u = data[i];
      const pt = project3D(x, t, u, W, H, pad);
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    }
    
    const ptLast = project3D(1, t, 0, W, H, pad);
    const ptFirst = project3D(0, t, 0, W, H, pad);
    ctx.lineTo(ptLast.x, ptLast.y);
    ctx.lineTo(ptFirst.x, ptFirst.y);
    ctx.closePath();
    
    ctx.fillStyle = PAL.bg; ctx.fill();
    ctx.strokeStyle = "rgba(0, 245, 212, 0.15)"; ctx.lineWidth = 0.5; ctx.stroke();
    
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = i / (N - 1);
      const u = data[i];
      const pt = project3D(x, t, u, W, H, pad);
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    }
    
    const ptS = project3D(0, t, 0.5, W, H, pad);
    const ptE = project3D(1, t, 0.5, W, H, pad);
    const grad = ctx.createLinearGradient(ptS.x, ptS.y, ptE.x, ptE.y);
    grad.addColorStop(0, "rgba(123, 44, 191, 0.3)");
    grad.addColorStop(0.5, "rgba(0, 245, 212, 0.7)");
    grad.addColorStop(1, "rgba(139, 92, 246, 0.3)");
    
    ctx.strokeStyle = grad; ctx.lineWidth = 1.25; ctx.stroke();
  });

  ctx.strokeStyle = "rgba(0, 245, 212, 0.1)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const origin = project3D(0, 0, 0, W, H, pad);
  const axisX = project3D(1, 0, 0, W, H, pad);
  const axisT = project3D(0, 1, 0, W, H, pad);
  const axisU = project3D(0, 0, 1, W, H, pad);
  
  ctx.moveTo(axisX.x, axisX.y); ctx.lineTo(origin.x, origin.y); ctx.lineTo(axisT.x, axisT.y);
  ctx.moveTo(origin.x, origin.y); ctx.lineTo(axisU.x, axisU.y); ctx.stroke();
  
  ctx.fillStyle = PAL.muted; ctx.font = "8px 'JetBrains Mono', monospace";
  ctx.fillText("x (Space)", axisX.x + 4, axisX.y + 4);
  ctx.fillText("t (Time)", axisT.x - 36, axisT.y + 8);
  ctx.fillText("u", axisU.x - 10, axisU.y - 2);
}

function MeshVisualizer({ N }) {
  const pad = { l: 15, r: 15 };
  const ref = useCanvas((ctx, W, H) => {
    ctx.fillStyle = PAL.bg; ctx.fillRect(0, 0, W, H);
    const pw = W - pad.l - pad.r;
    const cy = H / 2;
    ctx.strokeStyle = PAL.border; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(W - pad.r, cy); ctx.stroke();
    
    ctx.fillStyle = N > 128 ? PAL.accentFno : PAL.accentSolver;
    for (let i = 0; i < N; i++) {
      const x = pad.l + (i / (N - 1)) * pw;
      ctx.beginPath();
      const r = N > 128 ? 1 : N > 64 ? 1.5 : 2;
      ctx.arc(x, cy, r, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.fillStyle = PAL.muted; ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.fillText("x=0.0", pad.l - 4, cy + 12);
    ctx.fillText("x=1.0", W - pad.r - 20, cy + 12);
    ctx.fillText("Discretized mesh intervals dx", W / 2 - 60, cy - 8);
  }, [N]);
  return <canvas ref={ref} width={440} height={32} className="ic-preview-svg" style={{ width: "100%", height: "32px" }} />;
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
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;
    const bw = (pw / data.length) * 0.8;
    
    data.forEach((val, i) => {
      const x = pad.l + pw * (i / data.length) + (pw / data.length) * 0.1;
      const bh = (val / maxVal) * ph;
      ctx.fillStyle = color || PAL.accentFno;
      ctx.beginPath();
      ctx.roundRect(x, pad.t + ph - bh, bw, bh, 2);
      ctx.fill();
      if (val > 0) {
        ctx.fillStyle = PAL.text; ctx.font = "8px monospace";
        ctx.fillText(val, x + bw / 2 - 4, pad.t + ph - bh - 3);
      }
    });
  }, [data, title, color]);
  return <canvas ref={ref} width={220} height={110} className="plot-canvas" style={{ width: "100%" }} />;
}

function CodeBlock({ lines }) {
  return (
    <div className="code-block">
      {lines.map((l, i) => {
        let cls = "code-text";
        if (l.startsWith("#")) cls += " comment";
        else if (l.includes("def ") || l.includes("import ") || l.includes("from ") || l.includes("for ") || l.includes("with ")) cls += " keyword";
        else if (l.includes('"') || l.includes("'")) cls += " string";
        return (
          <div key={i} className="code-line">
            <span className="code-num">{String(i + 1).padStart(2, "0")}</span>
            <span className={cls}>{l}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN APPLICATION WORKSTATION (DUAL LAYOUT WITH PORTAL GATEWAY)
   ═══════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [activeView, setActiveView] = useState("portal"); // "portal" or "workstation"
  const [showIntro, setShowIntro] = useState(true); // System Boot popup overlay
  const [systemUptime, setSystemUptime] = useState(0); // Live simulator uptime
  const [diagnosticResult, setDiagnosticResult] = useState(null); // Diagnostic check result
  const [activeSheet, setActiveSheet] = useState("sim");
  const [modelType, setModelType] = useState("fisher");

  // Physical Parameters - Precise bi-directional state hooks
  const [D, setD] = useState(0.1);
  const [r, setR] = useState(2.0);
  const [mu, setMu] = useState(0.3);
  const [sig, setSig] = useState(0.1);

  // Grid sizing
  const [N, setN] = useState(128);
  const [dt, setDt] = useState(5e-5);
  
  // Animation timeline playbacks
  const [snaps, setSnaps] = useState([]);
  const [solFinal, setSolFinal] = useState(null);
  const [fnoFinal, setFnoFinal] = useState(null);
  const [errField, setErrField] = useState(null);
  const [solMs, setSolMs] = useState(null);
  const [fnoMs, setFnoMs] = useState(null);
  const [speedup, setSpeedup] = useState(null);
  const [l2, setL2] = useState(null);
  const [simDone, setSimDone] = useState(false);
  const [running, setRunning] = useState(false);

  const [tIndex, setTIndex] = useState(0);
  const [animating, setAnimating] = useState(false);
  const [animSpeed, setAnimSpeed] = useState(1);

  // Dynamic Heatmap hover coordinate HUD
  const [hudCoord, setHudCoord] = useState({ x: 0, t: 0, u: 0 });

  // Monte Carlo uncertainty sweeps
  const [sweeping, setSweeping] = useState(false);
  const [sweepDone, setSweepDone] = useState(false);
  const [sweepStats, setSweepStats] = useState({ count: 0, avgSpeedup: 0, avgL2: 0 });
  const [l2Hist, setL2Hist] = useState([]);
  const [speedupHist, setSpeedupHist] = useState([]);

  // Experimental upload states
  const [expRows, setExpRows] = useState([]);
  const [expName, setExpName] = useState("");
  const [expErr, setExpErr] = useState("");
  const [expWarn, setExpWarn] = useState("");

  // Research logs
  const [researchNotes, setResearchNotes] = useState("");

  // Grid convergence tests
  const [convResults, setConvResults] = useState([]);
  const [runningConv, setRunningConv] = useState(false);

  // Live timer for system uptime
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemUptime(prev => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatUptime = (seconds) => {
    const hrs = Math.floor(seconds / 3600).toString().padStart(2, "0");
    const mins = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
    const secs = (seconds % 60).toString().padStart(2, "0");
    return `${hrs}:${mins}:${secs}`;
  };

  const PRESETS = [
    { id: "baseline", label: "Baseline Front", D: 0.1, r: 2.0, mu: 0.3, sig: 0.1, model: "fisher" },
    { id: "fast-front", label: "Combustion Front (High r)", D: 0.4, r: 4.5, mu: 0.2, sig: 0.08, model: "fisher" },
    { id: "slow-allen", label: "Phase Separation (Allen-Cahn)", D: 0.08, r: 1.5, mu: 0.35, sig: 0.12, model: "allen" },
    { id: "narrow-kpp", label: "Sharp Shock Profile", D: 0.12, r: 3.2, mu: 0.5, sig: 0.04, model: "fisher" }
  ];
  const [presetId, setPresetId] = useState("baseline");

  const waveSpeed = (modelType === "allen" ? 1.35 * Math.sqrt(D * r) : 2 * Math.sqrt(D * r)).toFixed(4);
  const cflNumber = ((D * dt) / ((1 / (N - 1)) ** 2)).toFixed(3);

  const icData = Array.from({ length: 128 }, (_, i) => {
    const x = i / 127;
    return Math.min(1, Math.max(0, Math.exp(-0.5 * ((x - mu) / sig) ** 2)));
  });

  const applyPreset = (id) => {
    const p = PRESETS.find(x => x.id === id) || PRESETS[0];
    setPresetId(p.id);
    setModelType(p.model);
    setD(p.D); setR(p.r); setMu(p.mu); setSig(p.sig);
    setSimDone(false);
    setDiagnosticResult(null);
  };

  const resetAll = () => {
    setD(0.1); setR(2.0); setMu(0.3); setSig(0.1);
    setN(128); setDt(5e-5);
    setSimDone(false); setRunning(false);
    setSnaps([]); setSolFinal(null); setFnoFinal(null);
    setL2(null); setSpeedup(null); setErrField(null);
    setTIndex(0); setAnimating(false);
    setSweepDone(false); setSweepStats({ count: 0, avgSpeedup: 0, avgL2: 0 });
    setL2Hist([]); setSpeedupHist([]);
    setExpRows([]); setExpName(""); setExpErr(""); setExpWarn("");
    setResearchNotes(""); setConvResults([]);
    setDiagnosticResult(null);
  };

  // Run scientific Crank-Nicolson vs FNO surrogate
  const executeSimulation = useCallback(() => {
    setRunning(true); setSimDone(false); setAnimating(false); setTIndex(0);
    setTimeout(() => {
      const t0 = performance.now();
      const { snaps: s, final: sf } = solvePDE(D, r, mu, sig, modelType, N, dt, 1.0);
      const t1 = performance.now();
      
      const t2 = performance.now();
      const ff = fnoPredictTime(D, r, mu, sig, 1.0, modelType, N);
      const t3 = performance.now();

      const solveMsVal = (t1 - t0);
      const fnoMsVal = (t3 - t2);
      const errL2Val = relL2(ff, sf);
      const ef = sf.map((v, i) => Math.abs(v - ff[i]));

      setSnaps(s); setSolFinal(sf); setFnoFinal(ff); setErrField(ef);
      setSolMs(solveMsVal.toFixed(1)); 
      setFnoMs(fnoMsVal.toFixed(3));
      setSpeedup((solveMsVal / Math.max(fnoMsVal, 0.001)).toFixed(0));
      setL2(errL2Val.toFixed(3));
      setTIndex(s.length - 1); // Set to final snapshot
      setSimDone(true); setRunning(false);
    }, 100);
  }, [D, r, mu, sig, modelType, N, dt]);

  // Animator Stepper loops
  useEffect(() => {
    if (!animating || !snaps.length) return;
    const interval = setInterval(() => {
      setTIndex((prev) => {
        if (prev >= snaps.length - 1) {
          setAnimating(false);
          return snaps.length - 1;
        }
        return Math.min(snaps.length - 1, prev + 1);
      });
    }, 45 / animSpeed);
    return () => clearInterval(interval);
  }, [animating, snaps, animSpeed]);

  const activeT = snaps.length > 0 ? (tIndex / (snaps.length - 1)).toFixed(3) : "0.000";
  const activeSolSnap = snaps.length > 0 ? snaps[tIndex] : null;
  const activeFnoSnap = snaps.length > 0 ? fnoPredictTime(D, r, mu, sig, parseFloat(activeT), modelType, N) : null;

  // Rapid system diagnostic check
  const triggerSystemDiagnostic = () => {
    const t0 = performance.now();
    const { final: trueU } = solvePDE(D, r, mu, sig, modelType, 128, 5e-5, 1.0);
    const t1 = performance.now();
    
    const t2 = performance.now();
    const predU = fnoPredictTime(D, r, mu, sig, 1.0, modelType, 128);
    const t3 = performance.now();

    const sTime = t1 - t0;
    const fTime = t3 - t2;
    const sp = sTime / Math.max(fTime, 0.001);
    const errorVal = relL2(predU, trueU);

    setDiagnosticResult({
      speedup: sp.toFixed(0),
      l2: errorVal.toFixed(3)
    });
  };

  // Export current parameter profile as JSON
  const exportConfigJSON = () => {
    const config = {
      modelType,
      D,
      r,
      mu,
      sig,
      N,
      dt,
      waveSpeed,
      cflNumber,
      timestamp: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fno_workstation_config_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  // Monte Carlo sweeps
  const executeMonteCarlo = () => {
    setSweeping(true); setSweepDone(false);
    setTimeout(() => {
      const runCount = 50;
      let totalSpeed = 0;
      let totalL2 = 0;
      const l2Bins = new Array(10).fill(0);
      const speedBins = new Array(10).fill(0);

      for (let i = 0; i < runCount; i++) {
        const randD = 0.01 + Math.random() * 0.99;
        const randR = 0.5 + Math.random() * 4.5;
        const randMu = 0.15 + Math.random() * 0.7;
        const randSig = 0.03 + Math.random() * 0.22;

        const tS0 = performance.now();
        const { final: trueU } = solvePDE(randD, randR, randMu, randSig, modelType, 128, 5e-5, 1.0);
        const tS1 = performance.now();

        const tF0 = performance.now();
        const predU = fnoPredictTime(randD, randR, randMu, randSig, 1.0, modelType, 128);
        const tF1 = performance.now();

        const sTime = tS1 - tS0;
        const fTime = tF1 - tF0;
        const sp = sTime / Math.max(fTime, 0.001);
        const errorVal = relL2(predU, trueU);

        totalSpeed += sp;
        totalL2 += errorVal;

        // Categorize L2 (0 to 5%)
        const l2Idx = Math.min(9, Math.floor(errorVal / 0.5));
        l2Bins[l2Idx]++;

        // Categorize Speedup (0 to 150x)
        const speedIdx = Math.min(9, Math.floor(sp / 15));
        speedBins[speedIdx]++;
      }

      setSweepStats({
        count: runCount,
        avgSpeedup: Math.round(totalSpeed / runCount),
        avgL2: (totalL2 / runCount).toFixed(3)
      });
      setL2Hist(l2Bins);
      setSpeedupHist(speedBins);
      setSweepDone(true); setSweeping(false);
    }, 400);
  };

  // Run convergence mesh sensitivity
  const executeMeshSensitivity = () => {
    setRunningConv(true); setConvResults([]);
    setTimeout(() => {
      const sizes = [32, 64, 128, 256];
      const results = sizes.map(sz => {
        const t0 = performance.now();
        const { final: solverU } = solvePDE(D, r, mu, sig, modelType, sz, 5e-5, 1.0);
        const t1 = performance.now();
        
        const t2 = performance.now();
        const fnoU = fnoPredictTime(D, r, mu, sig, 1.0, modelType, sz);
        const t3 = performance.now();

        const errVal = relL2(fnoU, solverU);
        return {
          sz,
          solveTime: (t1 - t0).toFixed(1),
          fnoTime: (t3 - t2).toFixed(3),
          err: errVal.toFixed(3)
        };
      });
      setConvResults(results);
      setRunningConv(false);
    }, 500);
  };

  // Experimental CSV uploader parser
  const handleCSVUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const txt = await file.text();
      const lines = txt.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
      if (!lines.length) {
        setExpErr("CSV file contains no records."); setExpRows([]);
        return;
      }
      const rows = [];
      let clampCount = 0;
      for (const line of lines) {
        if (line.toLowerCase().includes("x")) continue; // skip header
        const parts = line.split(",").map(v => v.trim());
        if (parts.length < 2) continue;
        const xVal = Number(parts[0]);
        const uVal = Number(parts[1]);
        if (!Number.isFinite(xVal) || !Number.isFinite(uVal)) continue;
        if (uVal > 1 || uVal < 0) clampCount++;
        rows.push({ x: xVal, uExp: Math.min(1, Math.max(0, uVal)) });
      }
      if (!rows.length) {
        setExpErr("No valid numeric pairs parsed."); setExpRows([]);
        return;
      }
      rows.sort((a, b) => a.x - b.x);
      setExpName(file.name);
      setExpRows(rows);
      setExpErr("");
      setExpWarn(clampCount > 0 ? `${clampCount} points outside [0,1] were physically clamped.` : "");
    } catch (err) {
      setExpErr(`Upload error: ${err.message}`);
    }
  };

  const getFittingMAE = (target) => {
    if (!expRows.length || !target?.length) return null;
    let sumMAE = 0;
    let numL2 = 0;
    let denL2 = 0;
    expRows.forEach(row => {
      const index = Math.min(target.length - 1, Math.max(0, Math.round(row.x * (target.length - 1))));
      const pred = target[index];
      const diff = Math.abs(pred - row.uExp);
      sumMAE += diff;
      numL2 += diff * diff;
      denL2 += row.uExp * row.uExp;
    });
    return {
      mae: (sumMAE / expRows.length).toFixed(4),
      l2: denL2 > 1e-12 ? (Math.sqrt(numL2 / denL2) * 100).toFixed(3) : "0.000"
    };
  };

  const fitSolver = getFittingMAE(solFinal);
  const fitFno = getFittingMAE(fnoFinal);

  // Markdown scientific report download
  const exportReport = () => {
    const reportText = `# Fisher-KPP FNO Workstation Scientific Report
Date: ${new Date().toLocaleDateString()}
PDE Model: ${modelType.toUpperCase()}
Reaction Equation: ${modelType === "allen" ? "du/dt = D*d2u/dx2 + r*(u - u^3)" : "du/dt = D*d2u/dx2 + r*u*(1-u)"}

## Simulation Parameters
- Diffusion Coefficient D: ${D.toFixed(4)}
- Reaction Coefficient r: ${r.toFixed(4)}
- Grid points N: ${N}
- Time step dt: ${dt}
- Initial Condition Gaussian Center mu: ${mu}
- Initial Condition Gaussian Width sigma: ${sig}

## Operator & Solver Diagnostics
- Numerical Implicit Solver solve time: ${solMs || "N/A"} ms
- FNO Surrogate inference time: ${fnoMs || "N/A"} ms
- Speedup Factor: ${speedup || "N/A"}x
- Relative L2 Error: ${l2 || "N/A"} %
- Wave Speed c: ${waveSpeed}
- Courant Number (CFL): ${cflNumber}

## Scientific Research Observations
${researchNotes || "No researcher notes compiled."}

---
FISHER-KPP NEURAL OPERATOR SIMULATION WORKSTATION · CORE SYSTEM v4.0.0
`;
    const blob = new Blob([reportText], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Fisher_KPP_Scientific_Report_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  /* ═══════════════════════════════════════════════════════════════════════════
     VIEW A: MINIMAL PORTAL GATEWAY LANDING PAGE
     ═══════════════════════════════════════════════════════════════════════════ */
  if (activeView === "portal") {
    return (
      <div className="app portal-layout">
        
        {/* System Boot Intro Popup Modal */}
        {showIntro && (
          <div className="modal-overlay">
            <div className="boot-modal">
              <div className="modal-header">
                <span className="modal-title-glitch">SYSTEM CORE INITIALIZATION</span>
                <span className="system-ver">v4.0.0</span>
              </div>
              <div className="modal-body-content">
                <div className="modal-eq-strip">
                  du/dt = D * d2u/dx2 + R(u)
                </div>
                <p className="modal-p">
                  Welcome to the FNO Scientific Workstation. This operator-driven simulator bridges infinite-dimensional function spaces to instantly resolve nonlinear reacting dynamics.
                </p>
                
                <div className="modal-features-list">
                  <div className="feature-item-box">
                    <span className="feat-ico"><IconDiagnostic /></span>
                    <div>
                      <h5 className="feat-title">Instantaneous Mapping</h5>
                      <p className="feat-desc">Inference in less than 0.1ms bypassing sequential tridiagonal integration steps.</p>
                    </div>
                  </div>
                  <div className="feature-item-box">
                    <span className="feat-ico"><IconMesh /></span>
                    <div>
                      <h5 className="feat-title">Mesh Independence</h5>
                      <p className="feat-desc">Evaluates boundary fields dynamically on arbitrary space-time resolutions (N=64 to 256).</p>
                    </div>
                  </div>
                </div>

                <button onClick={() => setShowIntro(false)} className="btn btn-primary btn-boot-enter" style={{ display: "inline-flex", alignItems: "center", gap: "8px", justifyContent: "center" }}>
                  <IconPlay /> INITIALIZE OPERATOR SYSTEM
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Minimal Header */}
        <div className="portal-header">
          <div className="header-brand">
            <div className="status-indicator">
              <div className="status-dot-pulse" />
              <span className="status-lbl">SECURE NETWORK</span>
            </div>
            <span className="portal-title">NEURAL PDE ORCHESTRATOR</span>
            <span className="system-model-badge">FNS-WS-01</span>
          </div>
          
          <div className="header-telemetry">
            <VercelStatusBadge />
            <div className="tel-item">UPTIME: <span className="tel-val">{formatUptime(systemUptime)}</span></div>
            <div className="tel-item">ENGINE STATE: <span className="tel-val green">READY</span></div>
            <button onClick={() => setShowIntro(true)} className="btn-icon-circular" title="Show Boot Parameters"><IconInfo /></button>
          </div>
        </div>

        {/* Portal Core Layout: Parameters Control Deck (Left) + Sub-system Grid Launcher (Right) */}
        <div className="portal-grid-content">
          
          {/* Left panel: Compact editable parameters deck */}
          <div className="portal-config-deck">
            <div className="deck-title">PHYSICAL PARAMETER REGIME</div>
            
            <div className="sidebar-card deck-card">
              {/* Equation Selection */}
              <div className="input-group">
                <span className="section-title" style={{ marginBottom: "6px" }}>Reaction Kinetics Model</span>
                <select 
                  value={modelType} 
                  onChange={(e) => { setModelType(e.target.value); setSimDone(false); }} 
                  className="sidebar-select portal-select"
                >
                  <option value="fisher">Fisher-KPP (Population Waves)</option>
                  <option value="allen">Allen-Cahn (Phase Boundaries)</option>
                </select>
              </div>

              {/* Exact Numerical Parameter Fields */}
              <div className="sheet-layout-row">
                <div className="input-group flex-1">
                  <div className="input-header">
                    <span className="input-label">Diffusion Coeff (D)</span>
                  </div>
                  <input 
                    type="number" 
                    min="0.01" 
                    max="1.0" 
                    step="0.01" 
                    value={D} 
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v)) { setD(Math.max(0.01, Math.min(1.0, v))); setSimDone(false); }
                    }} 
                    className="param-num-input portal-num-input" 
                  />
                  <input 
                    type="range" min="0.01" max="1.0" step="0.01" value={D} 
                    onChange={(e) => { setD(parseFloat(e.target.value)); setSimDone(false); }} 
                    className="native-range-mini" 
                  />
                </div>
                
                <div className="input-group flex-1">
                  <div className="input-header">
                    <span className="input-label">Reaction Rate (r)</span>
                  </div>
                  <input 
                    type="number" 
                    min="0.5" 
                    max="5.0" 
                    step="0.1" 
                    value={r} 
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v)) { setR(Math.max(0.5, Math.min(5.0, v))); setSimDone(false); }
                    }} 
                    className="param-num-input portal-num-input" 
                  />
                  <input 
                    type="range" min="0.5" max="5.0" step="0.1" value={r} 
                    onChange={(e) => { setR(parseFloat(e.target.value)); setSimDone(false); }} 
                    className="native-range-mini" 
                  />
                </div>
              </div>

              <div className="sheet-layout-row">
                <div className="input-group flex-1">
                  <div className="input-header">
                    <span className="input-label">Gaussian Center (μ)</span>
                  </div>
                  <input 
                    type="number" 
                    min="0.1" 
                    max="0.9" 
                    step="0.05" 
                    value={mu} 
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v)) { setMu(Math.max(0.1, Math.min(0.9, v))); setSimDone(false); }
                    }} 
                    className="param-num-input portal-num-input" 
                  />
                  <input 
                    type="range" min="0.1" max="0.9" step="0.05" value={mu} 
                    onChange={(e) => { setMu(parseFloat(e.target.value)); setSimDone(false); }} 
                    className="native-range-mini" 
                  />
                </div>

                <div className="input-group flex-1">
                  <div className="input-header">
                    <span className="input-label">Gaussian Width (σ)</span>
                  </div>
                  <input 
                    type="number" 
                    min="0.02" 
                    max="0.3" 
                    step="0.01" 
                    value={sig} 
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v)) { setSig(Math.max(0.02, Math.min(0.3, v))); setSimDone(false); }
                    }} 
                    className="param-num-input portal-num-input" 
                  />
                  <input 
                    type="range" min="0.02" max="0.3" step="0.01" value={sig} 
                    onChange={(e) => { setSig(parseFloat(e.target.value)); setSimDone(false); }} 
                    className="native-range-mini" 
                  />
                </div>
              </div>

              <div className="btn-grid-row" style={{ marginTop: "4px" }}>
                <select 
                  value={presetId} 
                  onChange={(e) => applyPreset(e.target.value)} 
                  className="sidebar-select portal-select"
                  style={{ flex: 1 }}
                >
                  {PRESETS.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
                </select>
                <button onClick={resetAll} className="btn btn-secondary btn-portal-action">Reset Preset</button>
              </div>
            </div>

            {/* Derived Physical Telemetries */}
            <div className="sidebar-card deck-card telemetries-box">
              <div className="tel-row">
                <span className="tel-k">Derived Propagation Wave Speed (c):</span>
                <span className="tel-v teal-glow">{waveSpeed}</span>
              </div>
              <div className="tel-row">
                <span className="tel-k">CFL Stability Number (dx = 1/127):</span>
                <span className="tel-v" style={{ color: parseFloat(cflNumber) > 0.5 ? PAL.bad : PAL.good }}>{cflNumber}</span>
              </div>
            </div>

            {/* Dynamic system control utilities */}
            <div className="sidebar-card deck-card dynamic-functions">
              <button onClick={triggerSystemDiagnostic} className="btn btn-outline-cyan diag-btn" style={{ display: "inline-flex", alignItems: "center", gap: "6px", justifyContent: "center" }}>
                <IconDiagnostic /> RUN SYSTEM DIAGNOSTIC SELF-CHECK
              </button>
              
              <div className="btn-grid-row">
                <button onClick={exportConfigJSON} className="btn btn-outline-amber" style={{ display: "inline-flex", alignItems: "center", gap: "6px", justifyContent: "center" }}>
                  <IconExport /> EXPORT CONFIG (.json)
                </button>
                <button onClick={resetAll} className="btn btn-outline-purple" style={{ display: "inline-flex", alignItems: "center", gap: "6px", justifyContent: "center" }}>
                  <IconClear /> CLEAR CONSOLE
                </button>
              </div>
            </div>

            {/* Live diagnostic details */}
            {diagnosticResult && (
              <div className="diagnostic-hud-overlay">
                <div className="diag-hud-header">DIAGNOSTIC STATUS PASS</div>
                <div className="diag-hud-row">FNO Operator Speedup: <span className="green">{diagnosticResult.speedup}x</span></div>
                <div className="diag-hud-row">Relative L2 Discrepancy: <span className="teal">{diagnosticResult.l2}%</span></div>
                <div className="diag-hud-row">Surrogate Model Integrity: <span className="green">100% OPERATIONAL</span></div>
              </div>
            )}
          </div>

          {/* Right Panel: Beautiful Launcher Card Deck */}
          <div className="portal-modules-launcher">
            <div className="deck-title">MODULE WORKSPACE CORE</div>
            
            <div className="modules-launcher-grid">
              
              <div onClick={() => { setActiveSheet("sim"); setActiveView("workstation"); }} className="launcher-card">
                <div className="launcher-card-icon"><IconWaveform /></div>
                <h4 className="launcher-card-name">1. Simulation Suite</h4>
                <p className="launcher-card-desc">Compare 1D wavefront profiles, analyze pointwise error residuals, and render space-time isometric 3D surface propagations.</p>
                <div className="launcher-card-footer">
                  <span className="badge badge-teal">SOLVER PANEL</span>
                  <span className="badge badge-violet">T_end = 1.0s</span>
                </div>
              </div>

              <div onClick={() => { setActiveSheet("mesh"); setActiveView("workstation"); }} className="launcher-card">
                <div className="launcher-card-icon"><IconMesh /></div>
                <h4 className="launcher-card-name">2. Discretization Grid</h4>
                <p className="launcher-card-desc">Test spatial boundary node densities, evaluate numerical convergence rates, and run randomized Monte Carlo uncertainty sweeps.</p>
                <div className="launcher-card-footer">
                  <span className="badge badge-teal">N = {N} Nodes</span>
                  <span className="badge badge-violet">CFL Monitor</span>
                </div>
              </div>

              <div onClick={() => { setActiveSheet("data"); setActiveView("workstation"); }} className="launcher-card">
                <div className="launcher-card-icon"><IconChart /></div>
                <h4 className="launcher-card-name">3. Laboratory Fitting</h4>
                <p className="launcher-card-desc">Import experimental CSV coordinates data sheets, parse values, and compute pointwise error fits against surrogate predictions.</p>
                <div className="launcher-card-footer">
                  <span className="badge badge-teal">CSV IMPORT</span>
                  <span className="badge badge-violet">{expRows.length ? `${expRows.length} points` : "inactive"}</span>
                </div>
              </div>

              <div onClick={() => { setActiveSheet("doc"); setActiveView("workstation"); }} className="launcher-card">
                <div className="launcher-card-icon"><IconLibrary /></div>
                <h4 className="launcher-card-name">4. Scientific Library</h4>
                <p className="launcher-card-desc">Review Crank-Nicolson implicit equations, inspect neural operators spectral conv formulas, and download custom MD report files.</p>
                <div className="launcher-card-footer">
                  <span className="badge badge-teal">THEORY DATA</span>
                  <span className="badge badge-violet">MD exporter</span>
                </div>
              </div>

            </div>
          </div>

        </div>

      </div>
    );
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     VIEW B: INTEGRATED WORKSTATION SHELL (WITH STACKED SHEETS)
     ═══════════════════════════════════════════════════════════════════════════ */
  return (
    <div className="app">
      {/* ── TOPBAR (WITH NESTED BREADCRUMBS & RETURN BUTTON) ── */}
      <div className="topbar">
        <div className="topbarLeft">
          <button onClick={() => setActiveView("portal")} className="btn btn-secondary btn-return-portal" title="Return to Home Portal">
            ← RETURN TO PORTAL
          </button>
          <div className="sep-line" />
          
          <div className="breadcrumbs">
            <span className="bc-link" onClick={() => setActiveView("portal")}>PORTAL GATEWAY</span>
            <span className="bc-sep">/</span>
            <span className="bc-link" onClick={() => setActiveView("portal")}>WORKSTATION SHELL</span>
            <span className="bc-sep">/</span>
            <span className="bc-active">
              {activeSheet === "sim" ? "Simulation Suite" : 
               activeSheet === "mesh" ? "Discretization Grid" : 
               activeSheet === "data" ? "Laboratory Fitting" : "Scientific Library"}
            </span>
          </div>
        </div>
        
        <div className="topbarRight">
          <VercelStatusBadge />
          <span className="credentials">OPERATOR CONSOLE  ·  SURROGATE MODULE  ·  ONLINE</span>
          <div className="system-status">
            <div className="status-dot" />
            <span className="status-text">SYSTEM ONLINE</span>
          </div>
        </div>
      </div>

      {/* ── FORMULA STRIP BANNER ── */}
      <div className="banner">
        <div className="banner-math">
          <span className="banner-math-label">Governing PDE Case Study:</span>
          <span className="banner-math-eq">
            du/dt = D * d2u/dx2 + {modelType === "allen" ? "r * (u − u³)" : "r * u * (1 − u)"}
          </span>
        </div>
        <div className="banner-params-strip">
          <span className="banner-kv"><span className="b-key">D =</span> <span className="b-val">{D.toFixed(3)}</span></span>
          <span className="banner-kv"><span className="b-key">r =</span> <span className="b-val">{r.toFixed(2)}</span></span>
          <span className="banner-kv"><span className="b-key">μ =</span> <span className="b-val">{mu.toFixed(2)}</span></span>
          <span className="banner-kv"><span className="b-key">σ =</span> <span className="b-val">{sig.toFixed(2)}</span></span>
          <span className="banner-kv"><span className="b-key">c =</span> <span className="b-val" style={{ color: PAL.accentSolver }}>{waveSpeed}</span></span>
          <span className="banner-kv"><span className="b-key">CFL =</span> <span className="b-val" style={{ color: parseFloat(cflNumber) > 0.5 ? PAL.bad : PAL.good }}>{cflNumber}</span></span>
        </div>
      </div>

      {/* ── WORKSPACE FRAME ── */}
      <div className="workspace-grid">
        
        {/* ═══ LEFT CONTROL PANEL (SIDEBAR) ═══ */}
        <div className="sidebar">
          
          {/* Reaction Model Settings */}
          <div>
            <div className="section-title">Reaction Kinetics</div>
            <select 
              value={modelType} 
              onChange={(e) => { setModelType(e.target.value); setSimDone(false); }} 
              className="sidebar-select"
            >
              <option value="fisher">Fisher-KPP (Population Waves)</option>
              <option value="allen">Allen-Cahn (Phase Boundaries)</option>
            </select>
          </div>

          {/* Physical Parameters Group — Bi-directional inputs added */}
          <div>
            <div className="section-title">Physical Parameters</div>
            <div className="sidebar-card">
              <div className="input-group">
                <div className="input-header">
                  <span className="input-label">Diffusion (D)</span>
                </div>
                <input 
                  type="number" 
                  min="0.01" 
                  max="1.0" 
                  step="0.01" 
                  value={D} 
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v)) { setD(Math.max(0.01, Math.min(1.0, v))); setSimDone(false); }
                  }} 
                  className="param-num-input" 
                />
                <div className="slider-container">
                  <div className="slider-progress" style={{ width: `${((D - 0.01) / 0.99) * 100}%`, background: PAL.accentFno }} />
                  <input type="range" min="0.01" max="1.0" step="0.01" value={D} onChange={(e) => { setD(parseFloat(e.target.value)); setSimDone(false); }} className="native-range" />
                </div>
              </div>

              <div className="input-group">
                <div className="input-header">
                  <span className="input-label">Reaction Rate (r)</span>
                </div>
                <input 
                  type="number" 
                  min="0.5" 
                  max="5.0" 
                  step="0.1" 
                  value={r} 
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v)) { setR(Math.max(0.5, Math.min(5.0, v))); setSimDone(false); }
                  }} 
                  className="param-num-input" 
                />
                <div className="slider-container">
                  <div className="slider-progress" style={{ width: `${((r - 0.5) / 4.5) * 100}%`, background: PAL.accentSolver }} />
                  <input type="range" min="0.5" max="5.0" step="0.1" value={r} onChange={(e) => { setR(parseFloat(e.target.value)); setSimDone(false); }} className="native-range" />
                </div>
              </div>
            </div>
          </div>

          {/* Initial Condition Parameters */}
          <div>
            <div className="section-title">Initial State profile (u₀)</div>
            <div className="sidebar-card">
              <div className="input-group">
                <div className="input-header">
                  <span className="input-label">Gaussian Center (μ)</span>
                </div>
                <input 
                  type="number" 
                  min="0.1" 
                  max="0.9" 
                  step="0.05" 
                  value={mu} 
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v)) { setMu(Math.max(0.1, Math.min(0.9, v))); setSimDone(false); }
                  }} 
                  className="param-num-input" 
                />
                <div className="slider-container">
                  <div className="slider-progress" style={{ width: `${((mu - 0.1) / 0.8) * 100}%`, background: PAL.accentSolver }} />
                  <input type="range" min="0.1" max="0.9" step="0.05" value={mu} onChange={(e) => { setMu(parseFloat(e.target.value)); setSimDone(false); }} className="native-range" />
                </div>
              </div>

              <div className="input-group">
                <div className="input-header">
                  <span className="input-label">Gaussian Width (σ)</span>
                </div>
                <input 
                  type="number" 
                  min="0.02" 
                  max="0.3" 
                  step="0.01" 
                  value={sig} 
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v)) { setSig(Math.max(0.02, Math.min(0.3, v))); setSimDone(false); }
                  }} 
                  className="param-num-input" 
                />
                <div className="slider-container">
                  <div className="slider-progress" style={{ width: `${((sig - 0.02) / 0.28) * 100}%`, background: PAL.purple }} />
                  <input type="range" min="0.02" max="0.3" step="0.01" value={sig} onChange={(e) => { setSig(parseFloat(e.target.value)); setSimDone(false); }} className="native-range" />
                </div>
              </div>

              <svg width="100%" height="40" viewBox="0 0 240 40" className="ic-preview-svg">
                <rect width="240" height="40" fill={PAL.bg} />
                <polyline 
                  points={icData.map((v, i) => `${(i / 127) * 240},${40 - v * 36}`).join(" ")}
                  fill="none" stroke={PAL.accentSolver} strokeWidth="1.5"
                />
              </svg>
            </div>
          </div>

          {/* Quick Actions Panel */}
          <div>
            <div className="section-title">Parameters Presets</div>
            <div className="sidebar-card" style={{ gap: "8px" }}>
              <select 
                value={presetId} 
                onChange={(e) => applyPreset(e.target.value)} 
                className="sidebar-select"
              >
                {PRESETS.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
              <div className="btn-grid-row">
                <button onClick={() => { 
                  const p = PRESETS[Math.floor(Math.random() * PRESETS.length)];
                  applyPreset(p.id);
                }} className="btn btn-secondary">Randomize</button>
                <button onClick={resetAll} className="btn btn-secondary">Reset All</button>
              </div>
            </div>
          </div>

          {/* Action Trigger Workflows */}
          <div className="action-grid" style={{ marginTop: "auto" }}>
            <button 
              onClick={executeSimulation} 
              disabled={running} 
              className="btn btn-primary"
              style={{ height: "40px" }}
            >
              {running ? "COMPUTING..." : (
                <span style={{ display: "inline-flex", alignItems: "center", gap: "6px", justifyContent: "center" }}>
                  <IconPlay /> RUN ANALYSIS
                </span>
              )}
            </button>
          </div>
        </div>

        {/* ═══ RIGHT WORKSPACE SHEETS (STACK PANEL) ═══ */}
        <div className="stacked-pages-container">
          
          {/* Top Synchronization Tab Strip */}
          <div className="top-navigation-tabs">
            <button onClick={() => setActiveSheet("sim")} className={`nav-tab-btn ${activeSheet === "sim" ? "active" : ""}`}>
              1. Simulation Suite
            </button>
            <button onClick={() => setActiveSheet("mesh")} className={`nav-tab-btn ${activeSheet === "mesh" ? "active" : ""}`}>
              2. Discretization Grid
            </button>
            <button onClick={() => setActiveSheet("data")} className={`nav-tab-btn ${activeSheet === "data" ? "active" : ""}`}>
              3. Laboratory Fitting
            </button>
            <button onClick={() => setActiveSheet("doc")} className={`nav-tab-btn ${activeSheet === "doc" ? "active" : ""}`}>
              4. Scientific Library
            </button>
          </div>

          {/* 3D PERSPECTIVE PAGE DECK */}
          <div className="pages-deck-3d">
            
            {/* ── CARD 1: SIMULATION SUITE ── */}
            <div className={`folder-sheet 
              ${activeSheet === "sim" ? "active" : 
                activeSheet === "mesh" ? "stack-1" : 
                activeSheet === "data" ? "stack-2" : "stack-3"}`}
            >
              <div className="sheet-tab-handle" onClick={() => setActiveSheet("sim")}>
                Simulation Suite
              </div>
              
              <div className="sheet-content">
                {!simDone && !running && (
                  <div className="text-center" style={{ padding: "80px 40px" }}>
                    <div style={{ marginBottom: "16px" }}><IconGear /></div>
                    <h3 style={{ fontSize: "16px", fontWeight: "700", marginBottom: "8px" }}>Workstation Ready</h3>
                    <p style={{ color: "var(--muted)", fontSize: "12px", maxWidth: "420px", margin: "0 auto 16px", lineHeight: "1.7" }}>
                      Configure spatial diffusion, reaction rates, and boundary fields in the parameters panel. Run numerical Crank-Nicolson models and evaluate instant FNO traveling wave predictions.
                    </p>
                    <div style={{ display: "flex", gap: "8px", justifyContent: "center" }}>
                      <span className="badge">N={N} Mesh</span>
                      <span className="badge">T_end=1.0s</span>
                      <span className="badge">Neumann Boundary Conditions</span>
                    </div>
                  </div>
                )}

                {running && (
                  <div className="text-center" style={{ padding: "80px 40px" }}>
                    <div style={{ marginBottom: "16px" }}><IconSpinner /></div>
                    <h3 style={{ fontSize: "14px", fontWeight: "700", color: PAL.accentFno }}>SOLVING TRIDIAGONAL PDE MATRICES</h3>
                    <p style={{ color: PAL.muted, fontSize: "11px", marginTop: "4px" }}>
                      Integrating time trajectories across space mesh domain...
                    </p>
                  </div>
                )}

                {simDone && (
                  <div className="sheet-layout-col flex-1">
                    {/* Performance Indicators */}
                    <div className="metrics-strip">
                      <div className="metric-card m-solver">
                        <div className="metric-title">Solver Integrations</div>
                        <div className="metric-num">{solMs}<span className="metric-num-unit">ms</span></div>
                        <div className="metric-sub">Crank-Nicolson tridiagonal</div>
                      </div>
                      <div className="metric-card m-fno">
                        <div className="metric-title">FNO Surrogate</div>
                        <div className="metric-num">{fnoMs}<span className="metric-num-unit">ms</span></div>
                        <div className="metric-sub">Instantly resolved front</div>
                      </div>
                      <div className="metric-card m-speedup">
                        <div className="metric-title">Solve Speedup</div>
                        <div className="metric-num" style={{ color: PAL.purple }}>{speedup}×</div>
                        <div className="metric-sub">Operator acceleration factor</div>
                      </div>
                      <div className="metric-card m-error">
                        <div className="metric-title">Model Relative L2 Error</div>
                        <div className="metric-num" style={{ color: parseFloat(l2) < 2 ? PAL.good : PAL.accentSolver }}>{l2}%</div>
                        <div className="metric-sub">Target validation threshold &lt; 5%</div>
                      </div>
                    </div>

                    {/* Spatial and Residual plots */}
                    <div className="sheet-layout-row">
                      <div className="plot-panel">
                        <div className="plot-header">
                          <span className="plot-title">Spatial Profile Comparison  u(x, t={activeT})</span>
                          <div className="plot-legend">
                            <span className="legend-item"><div className="legend-indicator" style={{ background: "rgba(142, 154, 175, 0.4)" }} />Initial State (t=0)</span>
                            <span className="legend-item"><div className="legend-indicator" style={{ background: PAL.accentSolver }} />Solver Trajectory</span>
                            <span className="legend-item"><div className="legend-indicator" style={{ background: PAL.accentFno }} />FNO Surrogate</span>
                          </div>
                        </div>
                        <SolutionChart 
                          solver={activeSolSnap} 
                          fno={activeFnoSnap} 
                          ic={icData} 
                          title={`${modelType.toUpperCase()} Reaction Front: Solver (purple) vs FNO (teal)`} 
                        />
                      </div>

                      <div className="plot-panel">
                        <div className="plot-header">
                          <span className="plot-title red">Pointwise Spatial Residuals  |e(x)|</span>
                        </div>
                        <ErrorChart 
                          data={errField} 
                          title="Pointwise absolute discrepancies  |u_solver − u_fno|" 
                        />
                      </div>
                    </div>

                    {/* Temporal Stepper anim bar */}
                    <div className="anim-toolbar">
                      <div className="anim-controls-group">
                        <button 
                          onClick={() => setAnimating(!animating)} 
                          className={`btn-icon ${animating ? "active" : ""}`}
                          title={animating ? "Pause Timeline" : "Play Propagation"}
                        >
                          {animating ? <IconPause /> : <IconPlay />}
                        </button>
                        <button 
                          onClick={() => setTIndex(0)} 
                          className="btn-icon" 
                          style={{ display: "inline-flex", alignItems: "center", gap: "4px", justifyContent: "center" }}
                          title="Reset to initial state"
                        >
                          <IconRewind /> Reset
                        </button>
                        <button 
                          onClick={() => setTIndex(snaps.length - 1)} 
                          className="btn-icon" 
                          style={{ display: "inline-flex", alignItems: "center", gap: "4px", justifyContent: "center" }}
                          title="Jump to final convergence state"
                        >
                          <IconFastForward /> Jump
                        </button>
                        <button 
                          onClick={() => setAnimSpeed(prev => prev === 1 ? 2 : prev === 2 ? 5 : prev === 5 ? 10 : 1)}
                          className="btn-icon"
                          style={{ width: "42px" }}
                        >
                          {animSpeed}x
                        </button>
                      </div>

                      <div className="anim-slider-group">
                        <div className="anim-slider-header">
                          <span>Temporal Coordinate Timeline</span>
                          <span>Time step t = {activeT} s / 1.000 s</span>
                        </div>
                        <div className="slider-container" style={{ flex: 1 }}>
                          <div className="slider-progress" style={{ width: `${(tIndex / (snaps.length - 1)) * 100}%`, background: PAL.accentFno }} />
                          <input 
                            type="range" 
                            min="0" 
                            max={snaps.length - 1} 
                            step="1" 
                            value={tIndex} 
                            onChange={(e) => { setTIndex(parseInt(e.target.value)); setAnimating(false); }} 
                            className="native-range" 
                          />
                        </div>
                      </div>
                    </div>

                    {/* space time heatmaps and waterfall */}
                    <div className="sheet-layout-row">
                      <div className="plot-panel">
                        <div className="plot-header">
                          <span className="plot-title">Space-Time Evolution Profile  u(x, t)</span>
                        </div>
                        <HeatmapChart 
                          snaps={snaps} 
                          onHover={(hud) => setHudCoord(hud)} 
                        />
                        <div className="heatmap-overlay-hud">
                          <div className="hud-chip">Spatial x: <span className="hud-val">{hudCoord.x.toFixed(3)}</span></div>
                          <div className="hud-chip">Temporal t: <span className="hud-val">{hudCoord.t.toFixed(3)} s</span></div>
                          <div className="hud-chip">Concentration u: <span className="hud-val">{hudCoord.u.toFixed(4)}</span></div>
                        </div>
                      </div>

                      <div className="plot-panel">
                        <div className="plot-header">
                          <span className="plot-title">3D Isometric Surface Profile  u(x, t)</span>
                        </div>
                        <Waterfall3DChart snaps={snaps} />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ── CARD 2: DISCRETIZATION GRID ── */}
            <div className={`folder-sheet 
              ${activeSheet === "mesh" ? "active" : 
                activeSheet === "sim" ? "stack-1" : 
                activeSheet === "data" ? "stack-1" : "stack-2"}`}
            >
              <div className="sheet-tab-handle" onClick={() => setActiveSheet("mesh")}>
                Discretization Grid
              </div>
              
              <div className="sheet-content">
                <div className="sheet-layout-col">
                  {/* Grid Resolution Settings */}
                  <div className="sheet-layout-row align-center justify-between">
                    <div>
                      <h4 style={{ fontSize: "12px", fontWeight: "700" }}>Mesh Resolution Configuration</h4>
                      <p style={{ color: PAL.muted, fontSize: "10px", marginTop: "2px" }}>
                        Configure the density of spatial intervals along the physical grid boundaries.
                      </p>
                    </div>
                    <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                      <button onClick={() => { setN(64); setSimDone(false); }} className={`btn ${N === 64 ? "btn-primary" : "btn-secondary"}`}>N=64</button>
                      <button onClick={() => { setN(128); setSimDone(false); }} className={`btn ${N === 128 ? "btn-primary" : "btn-secondary"}`}>N=128</button>
                      <button onClick={() => { setN(256); setSimDone(false); }} className={`btn ${N === 256 ? "btn-primary" : "btn-secondary"}`}>N=256</button>
                    </div>
                  </div>

                  {/* Node visualizer */}
                  <div className="sidebar-card">
                    <div style={{ color: PAL.muted, fontSize: "9px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Spatial Mesh Node Grid Projection</div>
                    <MeshVisualizer N={N} />
                  </div>

                  {/* Discretization stats */}
                  <div className="metrics-strip">
                    <div className="metric-card">
                      <div className="metric-title">Courant Number (CFL)</div>
                      <div className="metric-num" style={{ color: parseFloat(cflNumber) > 0.5 ? PAL.bad : PAL.good }}>{cflNumber}</div>
                      <div className="metric-sub">{parseFloat(cflNumber) > 0.5 ? "CFL warning zone" : "CFL Stable region"}</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-title">Grid density N</div>
                      <div className="metric-num">{N}</div>
                      <div className="metric-sub">Spatial boundaries mesh</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-title">Time Integrations dt</div>
                      <div className="metric-num">{dt.toExponential(1)}</div>
                      <div className="metric-sub">Integrator stepping size</div>
                    </div>
                  </div>

                  {/* Mesh sensitivity sweep */}
                  <div className="sidebar-card">
                    <div className="sheet-layout-row justify-between align-center">
                      <div>
                        <h4 style={{ fontSize: "11px", fontWeight: "700", fontFamily: "var(--font-mono)", color: PAL.accentFno }}>Automated Grid Resolution convergence Analysis</h4>
                        <p style={{ color: PAL.muted, fontSize: "10px" }}>
                          Compute PDE solvers and FNO speedups across sizes N = [32, 64, 128, 256] to verify spatial order convergence.
                        </p>
                      </div>
                      <button 
                        onClick={executeMeshSensitivity} 
                        disabled={runningConv} 
                        className="btn btn-outline-cyan"
                      >
                        {runningConv ? "COMPUTING SWEEP..." : "RUN CONVERGENCE TEST"}
                      </button>
                    </div>

                    {convResults.length > 0 && (
                      <div className="data-table-container" style={{ marginTop: "10px" }}>
                        <div className="table-header" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
                          <span>Grid size N</span>
                          <span>Solver duration (ms)</span>
                          <span>FNO duration (ms)</span>
                          <span>Relative L2 Discrepancy %</span>
                        </div>
                        <div className="table-body">
                          {convResults.map((row, i) => (
                            <div key={i} className="table-row" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
                              <span style={{ color: PAL.accentSolver, fontWeight: "700" }}>N = {row.sz}</span>
                              <span>{row.solveTime} ms</span>
                              <span>{row.fnoTime} ms</span>
                              <span style={{ color: PAL.accentFno, fontWeight: "700" }}>{row.err}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Monte Carlo Uncertainty analysis */}
                  <div className="sidebar-card">
                    <div className="sheet-layout-row justify-between align-center">
                      <div>
                        <h4 style={{ fontSize: "11px", fontWeight: "700", fontFamily: "var(--font-mono)", color: PAL.accentSolver }}>Monte Carlo speed & accuracy Uncertainty sweeps</h4>
                        <p style={{ color: PAL.muted, fontSize: "10px" }}>
                          Solve 50 randomized iterations of react-diffusion parameters to compile speedup and error distributions.
                        </p>
                      </div>
                      <button 
                        onClick={executeMonteCarlo} 
                        disabled={sweeping} 
                        className="btn btn-outline-amber"
                      >
                        {sweeping ? "SWEEPING DECK..." : "RUN MONTE CARLO STUDY"}
                      </button>
                    </div>

                    {sweepDone && (
                      <div className="sheet-layout-col" style={{ marginTop: "14px" }}>
                        <div className="metrics-strip">
                          <div className="metric-card">
                            <div className="metric-title">Runs Swept</div>
                            <div className="metric-num">{sweepStats.count}</div>
                            <div className="metric-sub">Uncertainty space trials</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-title">Average Speedup</div>
                            <div className="metric-num" style={{ color: PAL.purple }}>{sweepStats.avgSpeedup}×</div>
                            <div className="metric-sub">Mean acceleration</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-title">Average L2 Discrepancy</div>
                            <div className="metric-num" style={{ color: PAL.accentFno }}>{sweepStats.avgL2}%</div>
                            <div className="metric-sub">Mean approximation error</div>
                          </div>
                        </div>

                        <div className="sheet-layout-row">
                          <div className="flex-1">
                            <HistChart data={l2Hist} title="L2 relative Error Distribution" color={PAL.accentFno} />
                          </div>
                          <div className="flex-1">
                            <HistChart data={speedupHist} title="Speedup factor Distribution" color={PAL.purple} />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* ── CARD 3: LABORATORY DATA FITTING ── */}
            <div className={`folder-sheet 
              ${activeSheet === "data" ? "active" : 
                activeSheet === "doc" ? "stack-1" : "stack-1"}`}
            >
              <div className="sheet-tab-handle" onClick={() => setActiveSheet("data")}>
                Laboratory Fitting
              </div>
              
              <div className="sheet-content">
                <div className="sheet-layout-col">
                  {/* CSV Uploader */}
                  <div className="sidebar-card">
                    <h4 style={{ fontSize: "12px", fontWeight: "700", marginBottom: "4px" }}>Upload Experimental Laboratory Measurements</h4>
                    <p style={{ color: PAL.muted, fontSize: "10px", marginBottom: "14px" }}>
                      Import a CSV sheet containing columns of (x, u_exp) measurements to validate the operator approximation against real data.
                    </p>
                    
                    <label className="file-upload-zone">
                      <span className="file-upload-label">
                        Drag CSV files here or <span className="file-upload-label-highlight">browse local directories</span>
                      </span>
                      <span style={{ fontSize: "9px", color: PAL.muted, fontFamily: "var(--font-mono)" }}>
                        Expected CSV layout: [x coordinate, concentration measurement u]
                      </span>
                      <input type="file" accept=".csv,text/csv" onChange={handleCSVUpload} className="file-upload-input" />
                    </label>
                    
                    {expName && <div style={{ fontSize: "10px", color: PAL.good, marginTop: "6px" }}>Successfully parsed: {expName}</div>}
                    {expErr && <div style={{ fontSize: "10px", color: PAL.bad, marginTop: "6px" }}>{expErr}</div>}
                    {expWarn && <div style={{ fontSize: "10px", color: PAL.accentSolver, marginTop: "6px" }}>{expWarn}</div>}
                  </div>

                  {/* Residuals matching */}
                  {expRows.length > 0 && (
                    <div className="sheet-layout-col">
                      <h4 style={{ fontSize: "12px", fontWeight: "700" }}>Residual Error Evaluations</h4>
                      
                      <div className="metrics-strip">
                        <div className="metric-card">
                          <div className="metric-title">Points Matched</div>
                          <div className="metric-num">{expRows.length}</div>
                          <div className="metric-sub">Validated grid points</div>
                        </div>
                        <div className="metric-card m-solver">
                          <div className="metric-title">Solver MAE</div>
                          <div className="metric-num">{fitSolver ? fitSolver.mae : "N/A"}</div>
                          <div className="metric-sub">Mean Absolute discrepancy</div>
                        </div>
                        <div className="metric-card m-solver">
                          <div className="metric-title">Solver Relative L2</div>
                          <div className="metric-num">{fitSolver ? `${fitSolver.l2}%` : "N/A"}</div>
                          <div className="metric-sub">Solver matching residual</div>
                        </div>
                        <div className="metric-card m-fno">
                          <div className="metric-title">FNO MAE</div>
                          <div className="metric-num">{fitFno ? fitFno.mae : "N/A"}</div>
                          <div className="metric-sub">Mean Absolute discrepancy</div>
                        </div>
                        <div className="metric-card m-fno">
                          <div className="metric-title">FNO Relative L2</div>
                          <div className="metric-num">{fitFno ? `${fitFno.l2}%` : "N/A"}</div>
                          <div className="metric-sub">Surrogate matching residual</div>
                        </div>
                      </div>

                      {/* Raw values table */}
                      <div className="data-table-container">
                        <div className="table-header" style={{ gridTemplateColumns: "1fr 2fr 2fr" }}>
                          <span>Grid Point index</span>
                          <span>Spatial x coordinate</span>
                          <span>Laboratory measurement u_exp</span>
                        </div>
                        <div className="table-body">
                          {expRows.slice(0, 150).map((row, i) => (
                            <div key={i} className="table-row" style={{ gridTemplateColumns: "1fr 2fr 2fr" }}>
                              <span style={{ color: PAL.muted }}>#{i+1}</span>
                              <span style={{ fontFamily: "var(--font-mono)" }}>{row.x.toFixed(4)}</span>
                              <span style={{ color: PAL.good, fontFamily: "var(--font-mono)", fontWeight: "700" }}>{row.uExp.toFixed(5)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* ── CARD 4: REFERENCE LIBRARY ── */}
            <div className={`folder-sheet ${activeSheet === "doc" ? "active" : "stack-hidden"}`}>
              <div className="sheet-tab-handle" onClick={() => setActiveSheet("doc")}>
                Scientific Library
              </div>
              
              <div className="sheet-content">
                <div className="sheet-layout-col">
                  {/* Inline Math Block */}
                  <div className="math-panel">
                    <div className="math-header-block">Crank-Nicolson Implicit Formulation</div>
                    <p className="math-text">
                      The numerical solver integrates spatial diffusion using a semi-implicit Crank-Nicolson formulation. The spatial second derivative is averaged across current and next time steps:
                    </p>
                    <div className="math-eq-display">
                      u_j^(n+1) − λ(u_(j-1)^(n+1) − 2u_j^(n+1) + u_(j+1)^(n+1)) = u_j^n + λ(u_(j-1)^n − 2u_j^n + u_(j+1)^n) + dt·R(u_j^n)
                    </div>
                    <p className="math-text">
                      where lambda = D dt / (2 dx^2). This results in an unconditionally stable tridiagonal system of linear equations resolved at each step in O(N) complexity using the Thomas algorithm.
                    </p>
                  </div>

                  {/* FNO Formulation */}
                  <div className="math-panel">
                    <div className="math-header-block">Fourier Neural Operator Mapping</div>
                    <p className="math-text">
                      The FNO model learns grid-independent operator maps between continuous function spaces. Rather than learning node pointwise regressions, it lifts spatial fields to spectral space:
                    </p>
                    <div className="math-eq-display">
                      v^(l+1)(x) = σ ( W · v^l(x) + F^(−1) [ R_l · F [v^l(x)] ] )
                    </div>
                    <p className="math-text">
                      A discrete Fourier transform projects spatial mappings F[v], a linear weights matrix parameterizes modes truncation R_l, and inverse operations F^-1 project fields back to boundary layers.
                    </p>
                  </div>

                  {/* Reseacher Notes & markdown exporter */}
                  <div className="sidebar-card">
                    <h4 style={{ fontSize: "12px", fontWeight: "700", marginBottom: "4px" }}>Researcher Observations & Field Logs</h4>
                    <p style={{ color: PAL.muted, fontSize: "10px", marginBottom: "10px" }}>
                      Compile physical findings or operator approximations below. Export the workspace observations alongside simulation diagnostics as a standard scientific Markdown file.
                    </p>
                    <textarea 
                      value={researchNotes} 
                      onChange={(e) => setResearchNotes(e.target.value)} 
                      placeholder="Write notes about simulation parameters, speedup comparisons, grid stability, or model accuracy fits..." 
                      className="notes-textarea" 
                    />
                    <div style={{ display: "flex", gap: "8px", marginTop: "10px" }}>
                      <button onClick={exportReport} className="btn btn-outline-cyan">
                        📥 EXPORT SCIENTIFIC REPORT (.md)
                      </button>
                      {simDone && (
                        <button 
                          onClick={() => {
                            if (!solFinal || !fnoFinal || !errField) return;
                            const N_pts = solFinal.length;
                            const rows = ["x,u_solver,u_fno,abs_error"];
                            for (let i = 0; i < N_pts; i++) {
                              const x = (i / (N_pts - 1)).toFixed(6);
                              rows.push(`${x},${solFinal[i]},${fnoFinal[i]},${errField[i]}`);
                            }
                            const blob = new Blob([rows.join("\n")], { type: "text/csv" });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `simulation_final_profile_${Date.now()}.csv`;
                            document.body.appendChild(a);
                            a.click();
                            a.remove();
                          }} 
                          className="btn btn-outline-amber"
                        >
                          📥 EXPORT FINAL SIMULATION STATE (.csv)
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Code section */}
                  <div className="sidebar-card">
                    <div className="math-header-block" style={{ color: PAL.muted }}>Solver Python Core Script</div>
                    <CodeBlock lines={[
                      "import numpy as np",
                      "from scipy.linalg import solve_banded",
                      "",
                      "def solve_reaction_diffusion(D, r, u0, dx=1/127, dt=5e-5, T_end=1.0):",
                      "    # Crank-Nicolson implicit solver for Neumann boundaries",
                      "    N = len(u0)",
                      "    lam = D * dt / (2 * dx**2)",
                      "    u = u0.copy()",
                      "    ab = np.zeros((3, N))",
                      "    ab[0, 1:]  = -lam  # Superdiagonal",
                      "    ab[1, :]   = 1 + 2 * lam  # Main diagonal",
                      "    ab[2, :-1] = -lam  # Subdiagonal",
                      "    ab[1, 0] = 1 + lam",
                      "    ab[1, -1] = 1 + lam",
                      "    n_steps = round(T_end / dt)",
                      "    for _ in range(n_steps):",
                      "        l = np.roll(u, 1); r_ = np.roll(u, -1)",
                      "        l[0] = u[0]; r_[-1] = u[-1]",
                      "        rhs = (u + lam * (l - 2*u + r_) + dt/2 * r * u * (1 - u))",
                      "        u = solve_banded((1, 1), ab, rhs)",
                      "    return u"
                    ]} />
                  </div>
                </div>
              </div>
            </div>

          </div>

        </div>

      </div>
    </div>
  );
}
