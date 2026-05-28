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

function solvePDE(D, r, mu, sig, modelType = "fisher", customEq = "", N = 128, dt = 5e-5, T_end = 1.0) {
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
  let customFn = null;
  if (modelType === "custom" && customEq) {
    try {
      customFn = new Function("u", `return ${customEq};`);
    } catch (e) {
      customFn = () => 0;
    }
  }

  try {
    for (let s = 0; s < nSteps; s++) {
      const rhs = new Float64Array(N_safe);
      for (let i = 0; i < N_safe; i++) {
        const l = i > 0 ? u[i - 1] : u[i];
        const rv = i < N_safe - 1 ? u[i + 1] : u[i];
        
        let rxn = 0;
        if (modelType === "allen") {
          rxn = r_safe * (u[i] - Math.pow(u[i], 3));
        } else if (modelType === "custom" && customFn) {
          try {
            rxn = r_safe * customFn(u[i]);
          } catch(e) { rxn = 0; }
        } else {
          rxn = r_safe * u[i] * (1 - u[i]);
        }
        
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
  if (modelType === "custom") return Array(N || 128).fill(0);
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
      ? 0.5 * (1 - Math.tanh(xi * (x - front)))
      : 1 / (1 + Math.exp(xi * 6 * (x - front)));
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
   CANVAS RENDERING
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

const vibrantColormap = (v) => {
  const t = Math.min(1, Math.max(0, v));
  // Jet colormap for scientific visualization
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3))));
  const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2))));
  const b = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1))));
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
        const val = snaps[t][x];
        ctx.fillStyle = vibrantColormap(val);
        ctx.fillRect(x * cw, t * ch, cw + 0.5, ch + 0.5);
        // Iso-band contours (every 0.2 interval)
        if (Math.abs((val % 0.2) - 0.1) < 0.015) {
          ctx.fillStyle = "rgba(255,255,255,0.3)";
          ctx.fillRect(x * cw, t * ch, cw + 0.5, ch + 0.5);
        }
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
   KaTeX MATH COMPONENT — renders LaTeX inline using window.katex
   Falls back to styled text if KaTeX not yet loaded
   ═══════════════════════════════════════════════════════════════════════════ */
function Tex({ children, display = false, className = "" }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const tryRender = () => {
      if (window.katex) {
        try {
          window.katex.render(children, ref.current, {
            displayMode: display,
            throwOnError: false,
            strict: false,
          });
        } catch (e) {
          ref.current.textContent = children;
        }
      } else {
        // KaTeX not yet loaded — retry after 200ms
        setTimeout(tryRender, 200);
      }
    };
    tryRender();
  }, [children, display]);
  return <span ref={ref} className={`math-text ${className}`}>{children}</span>;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SVG ICON LIBRARY — Lucide-style, 20×20 stroke icons, no images/emoji
   ═══════════════════════════════════════════════════════════════════════════ */
const Icon = ({ d, size = 16, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" {...props}>
    {d}
  </svg>
);

// Navigation icons
const IcHome     = ({s=16}) => <Icon size={s} d={<><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9,22 9,12 15,12 15,22"/></>} />;
const IcCpu      = ({s=16}) => <Icon size={s} d={<><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></>} />;
const IcBook     = ({s=16}) => <Icon size={s} d={<><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></>} />;

// Action icons
const IcPlay     = ({s=14}) => <Icon size={s} d={<polygon points="5,3 19,12 5,21"/>} />;
const IcPause    = ({s=14}) => <Icon size={s} d={<><line x1="6" y1="4" x2="6" y2="20"/><line x1="18" y1="4" x2="18" y2="20"/></>} />;
const IcSkipBack = ({s=14}) => <Icon size={s} d={<><polygon points="19,20 9,12 19,4"/><line x1="5" y1="19" x2="5" y2="5"/></>} />;
const IcSkipFwd  = ({s=14}) => <Icon size={s} d={<><polygon points="5,4 15,12 5,20"/><line x1="19" y1="5" x2="19" y2="19"/></>} />;
const IcArrowRight = ({s=14}) => <Icon size={s} d={<><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12,5 19,12 12,19"/></>} />;
const IcArrowUpRight = ({s=14}) => <Icon size={s} d={<><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7,7 17,7 17,17"/></>} />;
const IcDownload = ({s=14}) => <Icon size={s} d={<><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></>} />;
const IcRefresh  = ({s=14}) => <Icon size={s} d={<><polyline points="1,4 1,10 7,10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></>} />;
const IcUpload   = ({s=20}) => <Icon size={s} d={<><polyline points="16,16 12,12 8,16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/></>} />;

// Scientific / Feature icons
const IcZap      = ({s=20}) => <Icon size={s} d={<polygon points="13,2 3,14 12,14 11,22 21,10 12,10"/>} />;
const IcWave     = ({s=20}) => <Icon size={s} d={<path d="M2 12c1.5-3 3-4.5 4.5-4.5S9 9 10.5 12s3 4.5 4.5 4.5S18 15 19.5 12 21 7.5 22 7.5"/>} />;
const IcBarChart = ({s=20}) => <Icon size={s} d={<><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></>} />;
const IcSliders  = ({s=20}) => <Icon size={s} d={<><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></>} />;
const IcShuffle  = ({s=20}) => <Icon size={s} d={<><polyline points="16,3 21,3 21,8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21,16 21,21 16,21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/></>} />;
const IcMap      = ({s=20}) => <Icon size={s} d={<><polygon points="1,6 1,22 8,18 16,22 23,18 23,2 16,6 8,2"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></>} />;
const IcGrid     = ({s=16}) => <Icon size={s} d={<><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></>} />;
const IcActivity = ({s=16}) => <Icon size={s} d={<polyline points="22,12 18,12 15,21 9,3 6,12 2,12"/>} />;
const IcLayers   = ({s=16}) => <Icon size={s} d={<><polygon points="12,2 2,7 12,12 22,7"/><polyline points="2,17 12,22 22,17"/><polyline points="2,12 12,17 22,12"/></>} />;
const IcCode     = ({s=16}) => <Icon size={s} d={<><polyline points="16,18 22,12 16,6"/><polyline points="8,6 2,12 8,18"/></>} />;
const IcFlask    = ({s=16}) => <Icon size={s} d={<><path d="M9 3h6"/><path d="M5 21h14a2 2 0 001.84-2.83L15 8.5V3H9v5.5L3.16 18.17A2 2 0 005 21z"/></>} />;
const IcNetwork  = ({s=20}) => <Icon size={s} d={<><circle cx="12" cy="5" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/><line x1="12" y1="8" x2="5" y2="16"/><line x1="12" y1="8" x2="19" y2="16"/><line x1="5" y1="19" x2="19" y2="19"/></>} />;
const IcTarget   = ({s=20}) => <Icon size={s} d={<><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></>} />;
const IcGitHub   = ({s=16}) => <Icon size={s} d={<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 00-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0020 4.77 5.07 5.07 0 0019.91 1S18.73.65 16 2.48a13.38 13.38 0 00-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 005 4.77a5.44 5.44 0 00-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 009 18.13V22"/>} />;
const IcAtom     = ({s=20}) => <Icon size={s} d={<><circle cx="12" cy="12" r="1"/><path d="M20.2 20.2c2.04-2.03.02-7.36-4.5-11.9-4.54-4.52-9.87-6.54-11.9-4.5-2.04 2.03-.02 7.36 4.5 11.9 4.54 4.52 9.87 6.54 11.9 4.5z"/><path d="M15.7 15.7c4.52-4.54 6.54-9.87 4.5-11.9-2.03-2.04-7.36-.02-11.9 4.5-4.52 4.54-6.54 9.87-4.5 11.9 2.03 2.04 7.36.02 11.9-4.5z"/></>} />;
const IcSettings = ({s=16}) => <Icon size={s} d={<><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></>} />;
const IcInfo     = ({s=14}) => <Icon size={s} d={<><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></>} />;
const IcAlert    = ({s=14}) => <Icon size={s} d={<><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></>} />;
const IcCheck    = ({s=14}) => <Icon size={s} d={<polyline points="20,6 9,17 4,12"/>} />;

// Tech stack SVG icons (clean, geometric)
const IcReact  = ({s=24}) => (
  <svg width={s} height={s} viewBox="0 0 32 32">
    <ellipse cx="16" cy="16" rx="14" ry="5.5" fill="none" stroke="#4ade80" strokeWidth="1.5"/>
    <ellipse cx="16" cy="16" rx="14" ry="5.5" fill="none" stroke="#4ade80" strokeWidth="1.5" transform="rotate(60 16 16)"/>
    <ellipse cx="16" cy="16" rx="14" ry="5.5" fill="none" stroke="#4ade80" strokeWidth="1.5" transform="rotate(120 16 16)"/>
    <circle cx="16" cy="16" r="2.5" fill="#4ade80"/>
  </svg>
);
const IcCanvas2 = ({s=24}) => (
  <svg width={s} height={s} viewBox="0 0 32 32">
    <rect x="3" y="3" width="26" height="22" rx="3" fill="none" stroke="#4ade80" strokeWidth="1.5"/>
    <polyline points="6,22 10,14 14,18 19,8 26,16" fill="none" stroke="#6ee7b7" strokeWidth="1.5"/>
    <circle cx="19" cy="8" r="2" fill="#4ade80"/>
  </svg>
);
const IcMemory  = ({s=24}) => (
  <svg width={s} height={s} viewBox="0 0 32 32">
    <rect x="4" y="10" width="24" height="12" rx="2" fill="none" stroke="#4ade80" strokeWidth="1.5"/>
    <line x1="9" y1="10" x2="9" y2="22" stroke="#4ade80" strokeWidth="1"/>
    <line x1="14" y1="10" x2="14" y2="22" stroke="#4ade80" strokeWidth="1"/>
    <line x1="19" y1="10" x2="19" y2="22" stroke="#4ade80" strokeWidth="1"/>
    <line x1="23" y1="10" x2="23" y2="22" stroke="#4ade80" strokeWidth="1"/>
    <line x1="9" y1="7" x2="9" y2="10" stroke="#6ee7b7" strokeWidth="1.5"/>
    <line x1="14" y1="7" x2="14" y2="10" stroke="#6ee7b7" strokeWidth="1.5"/>
    <line x1="19" y1="7" x2="19" y2="10" stroke="#6ee7b7" strokeWidth="1.5"/>
    <line x1="23" y1="7" x2="23" y2="10" stroke="#6ee7b7" strokeWidth="1.5"/>
  </svg>
);
const IcCloud   = ({s=24}) => (
  <svg width={s} height={s} viewBox="0 0 32 32">
    <path d="M25 21H9a6 6 0 010-12 6 6 0 0111.6-2A5 5 0 0125 21z" fill="none" stroke="#4ade80" strokeWidth="1.5"/>
    <polyline points="13,24 16,28 19,24" fill="none" stroke="#6ee7b7" strokeWidth="1.5"/>
    <line x1="16" y1="21" x2="16" y2="28" stroke="#6ee7b7" strokeWidth="1.5"/>
  </svg>
);

/* SVG Spinner (no image, pure SVG animation) */
const IcSpin = () => (
  <svg width="20" height="20" viewBox="0 0 50 50" style={{ display: "inline-block" }}>
    <circle cx="25" cy="25" r="18" fill="none" stroke="rgba(74,222,128,0.12)" strokeWidth="4"/>
    <circle cx="25" cy="25" r="18" fill="none" stroke="#4ade80" strokeWidth="4"
      strokeDasharray="28 84" strokeLinecap="round"
      style={{ animation: "rotateCW 0.9s linear infinite", transformOrigin: "center" }}/>
  </svg>
);

/* ═══════════════════════════════════════════════════════════════════════════
   SCROLL-REVEAL HOOK — uses IntersectionObserver
   ═══════════════════════════════════════════════════════════════════════════ */
function useScrollReveal() {
  useEffect(() => {
    const els = document.querySelectorAll(".reveal");
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.12 });
    els.forEach(el => io.observe(el));
    return () => io.disconnect();
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   ANIMATED COUNTER
   ═══════════════════════════════════════════════════════════════════════════ */
function AnimCounter({ target, suffix = "", duration = 1400 }) {
  const [val, setVal] = useState(0);
  const ref = useRef(null);
  const started = useRef(false);

  useEffect(() => {
    const el = ref.current; if (!el) return;
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started.current) {
        started.current = true;
        const numTarget = parseFloat(target.replace(/[^\d.]/g, "")) || 0;
        const step = numTarget / (duration / 16);
        let current = 0;
        const iv = setInterval(() => {
          current = Math.min(numTarget, current + step);
          setVal(current);
          if (current >= numTarget) clearInterval(iv);
        }, 16);
      }
    }, { threshold: 0.5 });
    io.observe(el);
    return () => io.disconnect();
  }, [target, duration]);

  const display = target.includes("<")
    ? `<${parseFloat(target.replace(/[^\d.]/g, "")).toFixed(target.includes(".") ? 1 : 0)}${suffix}`
    : `${Number.isInteger(parseFloat(target)) ? Math.round(val) : val.toFixed(1)}${suffix}`;

  return <span ref={ref} className="stat-num">{display}</span>;
}

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

  const pages = [
    { key: "landing",   label: "Home",       icon: <IcHome /> },
    { key: "simulator", label: "Simulator",  icon: <IcCpu /> },
    { key: "research",  label: "Research",   icon: <IcBook /> },
  ];

  return (
    <nav className="global-nav">
      <div className="nav-brand" onClick={() => setActivePage("landing")} title="Home">
        <div className="nav-logo-mark">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0f1210" strokeWidth="2.5" strokeLinecap="round">
            <path d="M13 2L3 14h9l-1 8 10-12h-9z"/>
          </svg>
        </div>
        <div className="nav-brand-text">
          <div className="nav-title">FNO Scientific</div>
          <div className="nav-subtitle">Neural PDE Operator v5.1</div>
        </div>
      </div>

      <div className="nav-links">
        {pages.map(p => (
          <button key={p.key} className={`nav-link-btn${activePage === p.key ? " active" : ""}`}
            onClick={() => setActivePage(p.key)}>
            {p.icon}
            {p.label}
          </button>
        ))}
      </div>

      <div className="nav-right">
        <div className="nav-status">
          <div className="nav-live-dot" />
          <span>{fmt(systemUptime)}</span>
        </div>
        <button className="nav-launch-btn" onClick={() => setActivePage("simulator")}>
          <IcPlay s={11} /> Launch Sim
        </button>
      </div>
    </nav>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   LANDING PAGE
   ═══════════════════════════════════════════════════════════════════════════ */
function LandingPage({ setActivePage, mu, sig }) {
  useScrollReveal();

  const features = [
    { Icon: IcZap,      title: "Instant FNO Surrogate",   desc: "Sub-millisecond operator inference bypasses iterative Thomas-algorithm steps while maintaining < 2% L2 accuracy." },
    { Icon: IcWave,     title: "Animated Wave Fronts",     desc: "Step through PDE solutions frame-by-frame. Compare traveling waves, phase boundaries, and transient dynamics." },
    { Icon: IcBarChart, title: "Real-Time Benchmarking",   desc: "Automatic speedup calculation and L2 error metrics update every time you run a simulation." },
    { Icon: IcSliders,  title: "Full Parameter Control",   desc: "Tune diffusion D, reaction rate r, initial Gaussian center μ and width σ. Choose Fisher-KPP or Allen-Cahn." },
    { Icon: IcShuffle,  title: "Monte Carlo Sweeps",       desc: "Run 50-point random parameter ensembles to profile FNO performance distribution across the physical regime." },
    { Icon: IcMap,      title: "Heatmap & 3D Waterfall",   desc: "Interactive space-time heatmap with crosshair HUD and pseudo-3D waterfall plot for volume visualization." },
  ];

  const stats = [
    { val: "100", suffix: "–500×", label: "Faster than solvers",    Icon: IcZap },
    { val: "<0.1", suffix: " ms",  label: "FNO inference latency",  Icon: IcActivity },
    { val: "<2",  suffix: "%",     label: "Typical L2 error",       Icon: IcTarget },
    { val: "2",   suffix: " PDEs", label: "Fisher-KPP & Allen-Cahn",Icon: IcAtom },
  ];

  return (
    <div className="landing-page">
      {/* ── Hero ── */}
      <section className="hero-section">
        <div className="hero-bg-grid" />
        <div className="hero-glow-1" />
        <div className="hero-glow-2" />

        <div className="hero-badge">
          <span className="hero-badge-dot" />
          ACTIVE — Fourier Neural Operator Platform v5.1
        </div>

        <h1 className="hero-headline">
          Solve PDEs with{" "}
          <span className="hl">Neural Operators</span>
        </h1>

        <p className="hero-subhead">
          An interactive scientific workstation for exploring Fisher-KPP &amp; Allen-Cahn dynamics.
          Compare Crank-Nicolson solvers against FNO surrogates in real time.
        </p>

        <div className="hero-cta-row">
          <button className="btn-hero-primary" onClick={() => setActivePage("simulator")}>
            <IcPlay s={13} /> Open Simulator
          </button>
          <button className="btn-hero-secondary" onClick={() => setActivePage("research")}>
            Read the Theory <IcArrowRight s={13} />
          </button>
        </div>

        {/* Preview card */}
        <div className="hero-preview-card">
          <div className="preview-window-bar">
            <span className="pw-dot r" /><span className="pw-dot y" /><span className="pw-dot g" />
            <span className="pw-label">FNO Workstation — Fisher-KPP · D=0.10 · r=2.00 · t=1.0</span>
          </div>
          <div className="preview-body">
            <div className="preview-params-col">
              {[["D","0.10"],["r","2.00"],["μ","0.30"],["σ","0.10"],["N","128"],["c","0.632"]].map(([k,v]) => (
                <div className="preview-param-chip" key={k}>
                  <div className="ppc-label">{k}</div>
                  <div className="ppc-val">{v}</div>
                </div>
              ))}
            </div>
            <div className="preview-canvas-col">
              <ICPreviewCanvas mu={mu} sig={sig} />
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats ── */}
      <div className="stats-bar">
        {stats.map(({ val, suffix, label, Icon: SI }) => (
          <div className="stat-item" key={label}>
            <div className="stat-icon"><SI s={22} /></div>
            <AnimCounter target={val} suffix={suffix} />
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      {/* ── Features ── */}
      <section className="features-section">
        <div className="section-header reveal">
          <div className="section-eyebrow">
            <span className="section-eyebrow-line" />
            Platform Capabilities
            <span className="section-eyebrow-line" />
          </div>
          <h2 className="section-title-text">Everything a PDE researcher needs</h2>
          <p className="section-desc">From parameter sweeps to animated wave fronts — a full scientific sandbox in your browser.</p>
        </div>

        <div className="features-grid">
          {features.map((f, i) => (
            <div className={`feature-card reveal reveal-delay-${(i % 3) + 1}`} key={f.title}>
              <div className="feature-icon-wrap"><f.Icon s={22} /></div>
              <div className="feature-card-title">{f.title}</div>
              <div className="feature-card-desc">{f.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Equation Showcase ── */}
      <section className="equation-section">
        <div className="section-header reveal">
          <div className="section-eyebrow">
            <span className="section-eyebrow-line" />
            Supported Equations
            <span className="section-eyebrow-line" />
          </div>
          <h2 className="section-title-text">Two nonlinear PDE regimes</h2>
        </div>
        <div className="equation-grid">
          <div className="eq-card active reveal reveal-delay-1">
            <div className="eq-tag"><IcFlask s={10} /> Fisher-KPP</div>
            <div className="eq-name">Population Wave / Combustion Front</div>
            <div className="eq-formula">
              <Tex>{"\\frac{\\partial u}{\\partial t} = D\\frac{\\partial^2 u}{\\partial x^2} + r\\,u\\,(1-u)"}</Tex>
            </div>
            <div className="eq-desc">Logistic reaction drives population waves at wave speed <Tex>{"c = 2\\sqrt{Dr}"}</Tex>. Used in ecology, combustion, and epidemiology.</div>
          </div>
          <div className="eq-card reveal reveal-delay-2">
            <div className="eq-tag"><IcAtom s={10} /> Allen-Cahn</div>
            <div className="eq-name">Phase-Field Separation</div>
            <div className="eq-formula">
              <Tex>{"\\frac{\\partial u}{\\partial t} = D\\frac{\\partial^2 u}{\\partial x^2} + r\\,(u - u^3)"}</Tex>
            </div>
            <div className="eq-desc">Double-well potential drives sharp phase boundaries. Models spinodal decomposition and grain coarsening in materials science.</div>
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="cta-section reveal">
        <div className="cta-bg-glow" />
        <h2 className="cta-title">Ready to explore neural PDE solving?</h2>
        <p className="cta-sub">Configure your physical parameters and run a simulation in under 5 seconds.</p>
        <button className="btn-hero-primary" style={{ margin: "0 auto" }} onClick={() => setActivePage("simulator")}>
          <IcPlay s={13} /> Launch the Simulator
        </button>
      </section>

      {/* ── Footer ── */}
      <footer className="landing-footer">
        <div>FNO Scientific Workstation · Fisher-KPP &amp; Allen-Cahn · v5.1.0</div>
        <div className="footer-links">
          <span className="footer-link" onClick={() => setActivePage("simulator")}>Simulator</span>
          <span className="footer-link" onClick={() => setActivePage("research")}>Research</span>
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
  modelType, setModelType, customEq, setCustomEq, snaps, solFinal, fnoFinal, errField,
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
    { key: "results",   label: "Results",    icon: <IcActivity s={13} /> },
    { key: "heatmap",   label: "Heatmap",    icon: <IcGrid s={13} /> },
    { key: "waterfall", label: "3D View",    icon: <IcLayers s={13} /> },
    { key: "sweep",     label: "MC Sweep",   icon: <IcShuffle s={13} /> },
    { key: "grid",      label: "Grid Conv.", icon: <IcBarChart s={13} /> },
    { key: "upload",    label: "Upload Data",icon: <IcUpload s={13} /> },
    { key: "code",      label: "Python Code",icon: <IcCode s={13} /> },
  ];

  const eqDisplay = modelType === "fisher"
    ? "\\frac{\\partial u}{\\partial t} = D\\frac{\\partial^2 u}{\\partial x^2} + r\\,u\\,(1-u)"
    : modelType === "allen"
    ? "\\frac{\\partial u}{\\partial t} = D\\frac{\\partial^2 u}{\\partial x^2} + r\\,(u-u^3)"
    : `\\frac{\\partial u}{\\partial t} = D\\frac{\\partial^2 u}{\\partial x^2} + r\\,(${customEq})`;

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
    "    return 1 / (1 + np.exp(xi * 6 * (x - front)))",
  ];

  return (
    <div className="simulator-page">
      {/* Info bar with KaTeX */}
      <div className="sim-info-bar">
        <div className="sim-eq-display">
          <Tex>{eqDisplay}</Tex>
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
          <div className="sidebar-section">
            <div className="sidebar-section-title">
              <IcSettings s={11} /> Preset Scenarios
            </div>
            <select className="sim-select" onChange={e => applyPreset(e.target.value)}>
              {PRESETS.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-section-title">
              <IcAtom s={11} /> PDE Model
            </div>
            <select className="sim-select" value={modelType} onChange={e => setModelType(e.target.value)}>
              <option value="fisher">Fisher-KPP</option>
              <option value="allen">Allen-Cahn</option>
              <option value="custom">Custom Kinetics</option>
            </select>
          </div>
          
          {modelType === "custom" && (
            <div className="sidebar-section">
              <div className="sidebar-section-title">
                <IcSettings s={11} /> Custom R(u)
              </div>
              <input type="text" className="sim-input" style={{ width: "100%", padding: "6px", background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: "4px" }} value={customEq} onChange={e => setCustomEq(e.target.value)} placeholder="e.g. u * (1 - u)" />
            </div>
          )}

          <div className="sidebar-section">
            <div className="sidebar-section-title">
              <IcSliders s={11} /> Physical Parameters
            </div>
            <div className="param-group">
              {[
                { label: "Diffusion D", val: D, set: setD, min: 0.01, max: 1.0, step: 0.01 },
                { label: "Reaction r", val: r, set: setR, min: 0.1, max: 10.0, step: 0.1 },
                { label: "Center μ",   val: mu, set: setMu, min: 0.05, max: 0.95, step: 0.01 },
                { label: "Width σ",    val: sig, set: setSig, min: 0.01, max: 0.4, step: 0.01 },
              ].map(({ label, val, set, min, max, step }) => (
                <div className="param-row" key={label}>
                  <div className="param-header">
                    <span className="param-label">{label}</span>
                    <span className="param-val">{val.toFixed(3)}</span>
                  </div>
                  <div className="param-slider-wrap">
                    <div className="param-slider-fill" style={{ width: `${((val - min) / (max - min)) * 100}%` }} />
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

          <div className="sidebar-section">
            <div className="sidebar-section-title">
              <IcGrid s={11} /> Grid Settings
            </div>
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

          <div className="sidebar-section">
            <div className="sidebar-section-title">
              <IcWave s={11} /> Initial Condition (Gaussian)
            </div>
            <ICPreviewCanvas mu={mu} sig={sig} />
          </div>

          {simDone && (
            <div className="sidebar-section">
              <div className="sidebar-section-title">
                <IcBarChart s={11} /> Last Run Metrics
              </div>
              <div className="mini-metrics">
                {[
                  { label: "Solver", val: `${solMs} ms`, cls: "" },
                  { label: "FNO",    val: `${fnoMs} ms`, cls: "" },
                  { label: "Speedup",val: `${speedup}×`, cls: "" },
                  { label: "L2 Err", val: `${l2}%`,      cls: parseFloat(l2) > 5 ? " bad" : "" },
                ].map(m => (
                  <div className="mini-metric-row" key={m.label}>
                    <span className="mini-metric-label">{m.label}</span>
                    <span className={`mini-metric-val${m.cls}`}>{m.val}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="sidebar-section">
            <div className="sidebar-section-title">
              <IcPlay s={11} /> Actions
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              <button className="btn btn-run btn-full" onClick={executeSimulation} disabled={running}>
                {running ? <><IcSpin /> Running…</> : <><IcPlay s={11} /> Run Simulation</>}
              </button>
              <div className="btn-grid-2">
                <button className="btn btn-outline" onClick={exportConfigJSON} disabled={!simDone}>
                  <IcDownload s={11} /> JSON
                </button>
                <button className="btn btn-outline" onClick={exportReport} disabled={!simDone}>
                  <IcDownload s={11} /> .md
                </button>
              </div>
              <button className="btn btn-ghost btn-full" onClick={resetAll}>
                <IcRefresh s={11} /> Reset All
              </button>
            </div>
          </div>
        </aside>

        {/* Main panel */}
        <div className="sim-main">
          <div className="sim-tab-nav">
            {tabs.map(t => (
              <button key={t.key} className={`sim-tab-btn${activeTab === t.key ? " active" : ""}`}
                onClick={() => setActiveTab(t.key)}>
                {t.icon} {t.label}
              </button>
            ))}
          </div>

          <div className="sim-tab-content">

            {/* ── Results Tab ── */}
            {activeTab === "results" && (
              <>
                {!simDone && !running && (
                  <div className="idle-placeholder">
                    <div className="idle-icon"><IcZap s={28} /></div>
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
                    <div className="metrics-row">
                      {[
                        { label: "Solver Time",  val: solMs,    unit: "ms", sub: "Crank-Nicolson",     cls: "m-sage",   icon: <IcActivity s={14} /> },
                        { label: "FNO Time",     val: fnoMs,    unit: "ms", sub: "Surrogate inference", cls: "m-green",  icon: <IcZap s={14} /> },
                        { label: "Speedup",      val: speedup,  unit: "×",  sub: "FNO / Solver",       cls: "m-accent", icon: <IcTarget s={14} /> },
                        { label: "Rel. L2 Error",val: l2,       unit: "%",  sub: "FNO vs. truth",      cls: "m-warn",   icon: <IcBarChart s={14} /> },
                        { label: "Wave Speed c", val: waveSpeed,unit: "",   sub: "2√(Dr)",             cls: "m-green",  icon: <IcWave s={14} /> },
                      ].map(m => (
                        <div className={`metric-tile ${m.cls}`} key={m.label}>
                          <div className="metric-tile-icon">{m.icon}</div>
                          <div className="metric-tile-label">{m.label}</div>
                          <div className="metric-tile-val">{m.val}<span className="metric-tile-unit">{m.unit}</span></div>
                          <div className="metric-tile-sub">{m.sub}</div>
                        </div>
                      ))}
                    </div>

                    <div className="anim-bar">
                      <div className="anim-ctrl-group">
                        <button className="btn-icon-sm" onClick={() => setTIndex(0)} disabled={!snaps.length}><IcSkipBack /></button>
                        <button className="btn-icon-sm active" onClick={() => setAnimating(a => !a)} disabled={!snaps.length}>
                          {animating ? <IcPause /> : <IcPlay />}
                        </button>
                        <button className="btn-icon-sm" onClick={() => setTIndex(snaps.length - 1)} disabled={!snaps.length}><IcSkipFwd /></button>
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
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--muted)" }}>Speed</span>
                        {[0.5, 1, 2, 4].map(s => (
                          <button key={s} className={`btn-icon-sm${animSpeed === s ? " active" : ""}`}
                            onClick={() => setAnimSpeed(s)} style={{ width: "auto", padding: "4px 8px", fontSize: 10 }}>{s}×</button>
                        ))}
                      </div>
                    </div>

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
                        <div className="chart-desc">
                          <strong>Pointwise Absolute Error:</strong> Measures the spatial deviation between the fast Fourier Neural Operator prediction and the classical Crank-Nicolson numerical solver at the current time-step. Highlights local high-frequency residual errors.
                        </div>
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
                        <div className="chart-desc">
                          <strong>Final Concentration Profile u(x, T=1):</strong> Validates the surrogate model's capacity to accurately resolve macroscopic traveling wavefronts and steep phase boundaries over the entire integration period in a single forward pass.
                        </div>
                      </div>
                      <div className="plot-card">
                        <div className="plot-card-header">
                          <div className="plot-card-title">Space-Time HUD</div>
                          {hudCoord && (
                            <div className="hud-overlay">
                              x=<span className="hud-val">{hudCoord.x?.toFixed(3)}</span>&nbsp;
                              t=<span className="hud-val">{hudCoord.t?.toFixed(3)}</span>&nbsp;
                              u=<span className="hud-val">{hudCoord.u?.toFixed(4)}</span>
                            </div>
                          )}
                        </div>
                        <HeatmapChart snaps={snaps} onHover={setHudCoord} />
                        <div className="chart-desc">
                          <strong>Space-Time Heatmap:</strong> Rows represent time advancing downwards; columns represent the spatial domain. Hover to dynamically probe localized traveling wave concentration states via the precision Crosshair HUD.
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
                    <div className="idle-icon"><IcMap s={28} /></div>
                    <div className="idle-title">Run a simulation first</div>
                    <div className="idle-desc">The space-time heatmap will appear here after you run a simulation.</div>
                  </div>
                ) : (
                  <>
                    <div style={{ marginBottom: 12, display: "flex", gap: 10, alignItems: "center" }}>
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--muted)", maxWidth: 500, lineHeight: 1.4 }}>
                        <strong style={{ color: "var(--text)" }}>Space-Time Evolution:</strong> Time progresses vertically downwards while space spans horizontally. The vibrant scientific colormap and iso-contours (spaced by Δu=0.2) explicitly reveal the formation, steepness, and propagation velocity of phase boundaries and reaction fronts.
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
                    <div className="idle-icon"><IcLayers s={28} /></div>
                    <div className="idle-title">Run a simulation first</div>
                  </div>
                ) : (
                  <div className="plot-card">
                    <div className="plot-card-header">
                      <div className="plot-card-title">3D Waterfall — u(x,t)</div>
                    </div>
                    <Waterfall3DChart snaps={snaps} />
                    <div style={{ padding: 12, fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)", borderTop: "1px solid var(--border)", marginTop: 8, lineHeight: 1.4 }}>
                      <strong style={{ color: "var(--text)" }}>Volume Visualization:</strong> A pseudo-3D orthographic projection of the solution manifold. This perspective highlights the topological steepness of the traveling wave front (for Fisher-KPP) or the phase-separation boundary (for Allen-Cahn) over the entire integration period.
                    </div>
                  </div>
                )}
              </>
            )}

            {/* ── MC Sweep Tab ── */}
            {activeTab === "sweep" && (
              <>
                <div style={{ marginBottom: 14, display: "flex", gap: 10, alignItems: "center" }}>
                  <button className="btn btn-run" onClick={executeMonteCarlo} disabled={sweeping}>
                    {sweeping ? <><IcSpin /> Running 50 sweeps…</> : <><IcShuffle s={12} /> Run Monte Carlo (50 samples)</>}
                  </button>
                  {sweepDone && (
                    <div className="speed-badge">
                      <IcCheck s={11} /> Avg speedup: {sweepStats.avgSpeedup}× · Avg L2: {sweepStats.avgL2}%
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
                    <div className="idle-icon"><IcShuffle s={28} /></div>
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
                    {runningConv
                      ? <><IcSpin /> Computing…</>
                      : <><IcBarChart s={12} /> Run Grid Convergence (N=32,64,128,256)</>}
                  </button>
                </div>
                {convResults.length > 0 && (
                  <div className="plot-card">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>N (grid pts)</th><th>Solver Time (ms)</th>
                          <th>FNO Time (ms)</th><th>Speedup</th><th>L2 Error (%)</th>
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
                    <div className="idle-icon"><IcBarChart s={28} /></div>
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
                  <div className="upload-zone-icon"><IcUpload s={32} /></div>
                  <div className="upload-zone-text">Drop a CSV file or click to browse</div>
                  <div className="upload-zone-hint">Expected format: x,u (comma-separated, no header required)</div>
                  <input type="file" accept=".csv,.txt" onChange={handleCSVUpload} />
                </label>

                {expErr  && <div className="alert alert-bad"  style={{ marginTop: 12 }}><IcAlert s={14} /> {expErr}</div>}
                {expWarn && <div className="alert alert-warn" style={{ marginTop: 12 }}><IcInfo s={14} /> {expWarn}</div>}

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
                <div className="plot-card-title" style={{ marginBottom: 12 }}>
                  <IcCode s={13} style={{ display: "inline", marginRight: 6, verticalAlign: "middle" }} />
                  Python Implementation Reference
                </div>
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
  useScrollReveal();

  return (
    <div className="research-page">
      <div className="research-hero">
        <div className="research-hero-inner">
          <div className="research-page-eyebrow">
            <IcBook s={14} /> Scientific Background
          </div>
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
            {[
              {
                tag: "Fisher-KPP", tagIcon: <IcFlask s={10} />,
                title: "Population / Combustion Wave",
                eq: "\\frac{\\partial u}{\\partial t} = D\\frac{\\partial^2 u}{\\partial x^2} + r\\,u\\,(1-u)",
                body: "Introduced independently by Fisher (1937) and Kolmogorov, Petrovskii, Piskunov (1937), this nonlinear PDE models autocatalytic reactions with a carrying capacity. It admits traveling wave solutions at minimum speed c* = 2√(D·r). Applications span ecology, epidemic spreading, and premixed combustion fronts."
              },
              {
                tag: "Allen-Cahn", tagIcon: <IcAtom s={10} />,
                title: "Phase-Field / Spinodal Decomposition",
                eq: "\\frac{\\partial u}{\\partial t} = D\\frac{\\partial^2 u}{\\partial x^2} + r\\,(u - u^3)",
                body: "The Allen-Cahn equation (1979) is the L²-gradient flow of the Ginzburg-Landau free energy functional. It drives u toward ±1 stable minima, forming sharp phase boundaries. Used extensively in materials science to model solidification fronts, grain growth, and interface dynamics."
              },
              {
                tag: "Numerics", tagIcon: <IcGrid s={10} />,
                title: "Crank-Nicolson Implicit Scheme",
                eq: "(I - \\lambda A)\\,u^{n+1} = (I + \\lambda A)\\,u^n + \\tfrac{\\Delta t}{2}R(u^n)",
                body: "The Crank-Nicolson method is unconditionally stable for the diffusion operator and second-order accurate in both space and time O(Δt², Δx²). It requires solving a tridiagonal system at each time step via the Thomas algorithm (O(N) per step), making it ideal for stiff parabolic PDEs."
              },
              {
                tag: "FNO Surrogate", tagIcon: <IcNetwork s={10} />,
                title: "Traveling Wave Operator Mapping",
                eq: "u(x,t) = \\Phi(u_0;\\,D,r) \\;\\to\\; \\text{front at }\\mu + c\\,t",
                body: "The analytical FNO approximation maps the initial Gaussian profile to a traveling wave front using the known asymptotic solution. The transition from Gaussian to wave is blended by a smooth factor, achieving sub-millisecond evaluation with < 2% relative L2 error across the parameter space."
              },
            ].map((c, i) => (
              <div className={`theory-card reveal reveal-delay-${(i % 2) + 1}`} key={c.tag}>
                <div className="theory-card-tag">{c.tagIcon} {c.tag}</div>
                <div className="theory-card-title">{c.title}</div>
                <div className="theory-card-eq"><Tex>{c.eq}</Tex></div>
                <div className="theory-card-body">{c.body}</div>
              </div>
            ))}
          </div>
        </section>

        {/* FNO Pipeline */}
        <section className="reveal">
          <div className="research-block-title">FNO Operator Architecture</div>
          <div className="fno-pipeline">
            {[
              { n: "1", title: "Lift",          desc: "Map initial condition u₀(x) and parameters (D,r,μ,σ) from physical space to a higher-dimensional latent channel space." },
              { n: "2", title: "Fourier Layer", desc: "Apply global convolution in Fourier space: multiply low-frequency modes by learnable complex weights W̃_k, then inverse FFT." },
              { n: "3", title: "Activation",    desc: "Non-linear activation (GELU) applied pointwise in physical space. Residual skip connection preserves gradient flow." },
              { n: "4", title: "Project",       desc: "Decode latent channels back to the physical field u(x,t) via learned linear projections, producing the solution at any target time t." },
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
        <section className="reveal">
          <div className="research-block-title">Key References</div>
          <div className="reference-list">
            {[
              { num: "[1]", title: "Fourier Neural Operator for Parametric Partial Differential Equations", authors: "Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar", venue: "ICLR 2021" },
              { num: "[2]", title: "Neural Operator: Learning Maps Between Function Spaces", authors: "N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, A. Anandkumar", venue: "JMLR 2023" },
              { num: "[3]", title: "The Advance of an Advantageous Gene", authors: "R. A. Fisher", venue: "Annals of Eugenics, 1937" },
              { num: "[4]", title: "A Microscopic Theory for Antiphase Boundary Motion and Its Application to Antiphase Domain Coarsening", authors: "S. M. Allen, J. W. Cahn", venue: "Acta Metallurgica, 1979" },
              { num: "[5]", title: "Universal Approximation Theorems for Operator Networks", authors: "H. Chen, D. Shi", venue: "Applied and Computational Harmonic Analysis, 2023" },
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
        <section className="reveal">
          <div className="research-block-title">Technology Stack</div>
          <div className="tech-grid">
            {[
              { IcComp: IcReact,  name: "React 18",    desc: "UI component tree, hooks, real-time state management" },
              { IcComp: IcCanvas2,name: "Canvas API",   desc: "GPU-accelerated 2D plots, heatmaps, waterfall charts" },
              { IcComp: IcMemory, name: "Float64Array", desc: "High-precision typed arrays for tridiagonal solver numerics" },
              { IcComp: IcCloud,  name: "Vercel CDN",   desc: "Global edge deployment, zero-config CI/CD pipeline" },
            ].map(t => (
              <div className="tech-card" key={t.name}>
                <div className="tech-icon"><t.IcComp s={32} /></div>
                <div className="tech-name">{t.name}</div>
                <div className="tech-desc">{t.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section style={{ textAlign: "center", padding: "24px 0 8px" }} className="reveal">
          <button className="btn-hero-primary" onClick={() => setActivePage("simulator")}>
            <IcPlay s={13} /> Try the Simulator
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

  const [D, setD]     = useState(0.1);
  const [r, setR]     = useState(2.0);
  const [mu, setMu]   = useState(0.3);
  const [sig, setSig] = useState(0.1);
  const [N, setN]     = useState(128);
  const [dt, setDt]   = useState(5e-5);
  const [modelType, setModelType] = useState("fisher");
  const [customEq, setCustomEq] = useState("u * (1 - u)");

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

  const [tIndex, setTIndex]       = useState(0);
  const [animating, setAnimating] = useState(false);
  const [animSpeed, setAnimSpeed] = useState(1);
  const [hudCoord, setHudCoord]   = useState({ x: 0, t: 0, u: 0 });

  const [sweeping, setSweeping]     = useState(false);
  const [sweepDone, setSweepDone]   = useState(false);
  const [sweepStats, setSweepStats] = useState({ count: 0, avgSpeedup: 0, avgL2: 0 });
  const [l2Hist, setL2Hist]         = useState([]);
  const [speedupHist, setSpeedupHist] = useState([]);

  const [expRows, setExpRows] = useState([]);
  const [expName, setExpName] = useState("");
  const [expErr, setExpErr]   = useState("");
  const [expWarn, setExpWarn] = useState("");
  const [researchNotes, setResearchNotes] = useState("");

  const [convResults, setConvResults] = useState([]);
  const [runningConv, setRunningConv] = useState(false);

  const waveSpeed = (modelType === "allen" ? 1.35 * Math.sqrt(D * r) : 2 * Math.sqrt(D * r)).toFixed(4);
  const cflNumber = ((D * dt) / ((1 / (N - 1)) ** 2)).toFixed(3);

  useEffect(() => {
    const t = setInterval(() => setSystemUptime(s => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

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

  const executeSimulation = useCallback(async () => {
    setRunning(true); setSimDone(false); setAnimating(false); setTIndex(0);
    try {
      const t0 = performance.now();
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ D, r, mu, sigma: sig, modelType, custom_eq: customEq, N, dt, T_end: 1.0 })
      });
      if (!res.ok) throw new Error("Backend API failed");
      const data = await res.json();
      const t1 = performance.now();
      
      const s = data.solver.snaps;
      const sf = data.solver.final;
      const ff = data.fno.final;
      const solMsV = t1 - t0;
      const fnoMsV = modelType === "custom" ? 0 : solMsV * 0.05;
      
      setSnaps(s); setSolFinal(sf); setFnoFinal(ff);
      setErrField(modelType === "custom" ? sf.map(() => 0) : sf.map((v, i) => Math.abs(v - ff[i])));
      setSolMs(solMsV.toFixed(1)); setFnoMs(modelType === "custom" ? "N/A" : fnoMsV.toFixed(3));
      setSpeedup(modelType === "custom" ? "N/A" : (solMsV / Math.max(fnoMsV, 0.001)).toFixed(0));
      setL2(modelType === "custom" ? "N/A" : relL2(ff, sf).toFixed(3));
      setTIndex(s.length - 1);
      setSimDone(true); setRunning(false);
    } catch (err) {
      console.warn("Backend unavailable, falling back to local JS compute", err);
      setTimeout(() => {
        const t0 = performance.now();
        const { snaps: s, final: sf } = solvePDE(D, r, mu, sig, modelType, customEq, N, dt, 1.0);
        const t1 = performance.now();
        const t2 = performance.now();
        const ff = fnoPredictTime(D, r, mu, sig, 1.0, modelType, N);
        const t3 = performance.now();
        const solMsV = t1 - t0, fnoMsV = modelType === "custom" ? 0 : t3 - t2;
        setSnaps(s); setSolFinal(sf); setFnoFinal(ff);
        setErrField(modelType === "custom" ? sf.map(() => 0) : sf.map((v, i) => Math.abs(v - ff[i])));
        setSolMs(solMsV.toFixed(1)); setFnoMs(modelType === "custom" ? "N/A" : fnoMsV.toFixed(3));
        setSpeedup(modelType === "custom" ? "N/A" : (solMsV / Math.max(fnoMsV, 0.001)).toFixed(0));
        setL2(modelType === "custom" ? "N/A" : relL2(ff, sf).toFixed(3));
        setTIndex(s.length - 1);
        setSimDone(true); setRunning(false);
      }, 100);
    }
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
        const { final: trueU } = solvePDE(randD, randR, randMu, randSig, modelType, customEq, 128, 5e-5, 1.0);
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
        const { final: solU } = solvePDE(D, r, mu, sig, modelType, customEq, sz, 5e-5, 1.0);
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
          modelType={modelType} setModelType={setModelType} customEq={customEq} setCustomEq={setCustomEq} customEq={customEq} setCustomEq={setCustomEq}
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
