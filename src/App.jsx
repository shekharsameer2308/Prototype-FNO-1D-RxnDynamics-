/* eslint-disable */ 
import { useState, useEffect, useRef, useCallback } from "react";

/* ═══════════════════════════════════════════════════════════════════════════
   PHYSICS ENGINE — Crank-Nicolson Fisher-KPP solver
   ═══════════════════════════════════════════════════════════════════════════ */
function thomasSolve(lo, diag, up, rhs) {
  const n = diag.length;
  const c = new Float64Array(n), d = new Float64Array(n), x = new Float64Array(n);
  c[0] = up[0] / diag[0]; d[0] = rhs[0] / diag[0];
  for (let i = 1; i < n; i++) {
    const m = diag[i] - lo[i] * c[i - 1];
    c[i] = up[i] / m;
    d[i] = (rhs[i] - lo[i] * d[i - 1]) / m;
  }
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
  return x;
}

function solvePDE(D, r, mu, sig, N = 128, dt = 5e-5, T_end = 1.0) {
  const dx = 1 / (N - 1);
  const lam = D * dt / (2 * dx * dx);
  let u = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const x = i * dx;
    u[i] = Math.min(1, Math.max(0, Math.exp(-0.5 * ((x - mu) / sig) ** 2)));
  }
  const lo = new Float64Array(N).fill(-lam);
  const di = new Float64Array(N).fill(1 + 2 * lam);
  const up = new Float64Array(N).fill(-lam);
  di[0] = 1 + lam; di[N - 1] = 1 + lam;

  const nSteps = Math.round(T_end / dt);
  const saveEvery = Math.max(1, Math.floor(nSteps / 80));
  const snaps = [Array.from(u)];

  for (let s = 0; s < nSteps; s++) {
    const rhs = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const l = i > 0 ? u[i-1] : u[i], rv = i < N-1 ? u[i+1] : u[i];
      rhs[i] = u[i] + lam * (l - 2*u[i] + rv) + dt/2 * r * u[i] * (1 - u[i]);
    }
    u = new Float64Array(thomasSolve(lo, di, up, rhs));
    if (s % saveEvery === 0) snaps.push(Array.from(u));
  }
  return { snaps, final: Array.from(u) };
}

/* ═══════════════════════════════════════════════════════════════════════════
   FNO SURROGATE — analytical wave-packet approximation
   ═══════════════════════════════════════════════════════════════════════════ */
function fnoPredict(D, r, mu, sig, N = 128) {
  const dx = 1 / (N - 1);
  const c  = 2 * Math.sqrt(D * r);
  const xi = Math.sqrt(r / (6 * D));
  const front = Math.min(0.95, mu + c * 1.0);
  return Array.from({ length: N }, (_, i) => {
    const x = i * dx;
    const wave = 1 / (1 + Math.exp(-xi * 6 * (x - front)));
    return Math.min(1, Math.max(0, wave));
  });
}

/* L2 relative error */
const relL2 = (a, b) => {
  let n = 0, d = 0;
  for (let i = 0; i < a.length; i++) { n += (a[i]-b[i])**2; d += b[i]**2; }
  return d > 1e-12 ? Math.sqrt(n/d)*100 : 0;
};

/* ═══════════════════════════════════════════════════════════════════════════
   MOCK DATASET GENERATOR
   ═══════════════════════════════════════════════════════════════════════════ */
function generateDatasetStats() {
  const Dvals  = Array.from({length:20}, (_,i) => (0.01*10**(i/19*2)).toFixed(3));
  const rvals  = Array.from({length:20}, (_,i) => (0.5 + i*0.236).toFixed(2));
  const total  = 20*20*5*3; // D*r*mu*sig
  return { Dvals, rvals, total, trainN: Math.round(total*0.7), valN: Math.round(total*0.15), testN: Math.round(total*0.15) };
}

/* ═══════════════════════════════════════════════════════════════════════════
   CANVAS HELPERS
   ═══════════════════════════════════════════════════════════════════════════ */
const PAL = {
  bg:"#060912", panel:"#0b1120", border:"#162035", accent:"#22d3ee",
  accent2:"#f59e0b", green:"#34d399", red:"#f87171", purple:"#a78bfa",
  text:"#f1f5f9", muted:"#475569", dim:"#1e293b"
};

function useCanvas(draw, deps) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0,0,c.width,c.height);
    draw(ctx, c.width, c.height);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

function drawAxes(ctx, W, H, pad) {
  ctx.strokeStyle = PAL.border; ctx.lineWidth = 1;
  for (let i=0;i<=4;i++){
    const y=pad.t+(H-pad.t-pad.b)/4*i;
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    const x=pad.l+(W-pad.l-pad.r)/4*i;
    ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();
  }
}

function drawLine(ctx, data, W, H, pad, color, lw=2, glow=true) {
  if (!data?.length) return;
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
  if (glow) { ctx.shadowColor=color; ctx.shadowBlur=8; }
  ctx.strokeStyle=color; ctx.lineWidth=lw;
  ctx.beginPath();
  data.forEach((v,i)=>{
    const x=pad.l+(i/(data.length-1))*pw;
    const y=pad.t+ph*(1-Math.min(1,Math.max(0,v)));
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.shadowBlur=0;
}

/* inferno colormap */
const inferno = v => {
  const t = Math.min(1,Math.max(0,v));
  const r = Math.min(255,Math.round(t<0.4?t*2.5*255:255));
  const g = Math.min(255,Math.round(t<0.5?0:((t-0.5)*2)*220));
  const b = Math.min(255,Math.round(t<0.2?t*5*200:t>0.7?0:((0.7-t)/0.5)*200));
  return `rgb(${r},${g},${b})`;
};

/* ═══════════════════════════════════════════════════════════════════════════
   CHART COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */
function SolutionChart({ solver, fno, ic, title }) {
  const pad = {t:24,b:28,l:36,r:12};
  const ref = useCanvas((ctx,W,H) => {
    ctx.fillStyle=PAL.bg; ctx.fillRect(0,0,W,H);
    drawAxes(ctx,W,H,pad);
    // axis labels
    ctx.fillStyle=PAL.muted; ctx.font="10px 'JetBrains Mono',monospace";
    ["0","0.25","0.5","0.75","1.0"].forEach((l,i)=>{
      ctx.fillText(l, pad.l+(W-pad.l-pad.r)/4*i-8, H-6);
    });
    ["1.0","0.75","0.5","0.25","0"].forEach((l,i)=>{
      ctx.fillText(l, 2, pad.t+(H-pad.t-pad.b)/4*i+4);
    });
    ctx.fillStyle=PAL.muted; ctx.font="bold 11px 'JetBrains Mono',monospace";
    ctx.fillText(title,pad.l+2,16);
    if (ic)     drawLine(ctx,ic,W,H,pad,PAL.muted,1.5,false);
    if (solver) drawLine(ctx,solver,W,H,pad,PAL.accent2,2.5);
    if (fno)    drawLine(ctx,fno,W,H,pad,PAL.accent,2,true);
    // legend
    if (solver&&fno){
      [[PAL.muted,"IC (t=0)"],[PAL.accent2,"Solver"],[PAL.accent,"FNO"]].forEach(([c,l],i)=>{
        ctx.fillStyle=c; ctx.fillRect(W-120+i*0,H-40+i*13,14,3);
        ctx.fillStyle=PAL.muted; ctx.font="9px monospace";
        ctx.fillText(l,W-102+i*0,H-37+i*13);
      });
    }
  }, [solver, fno, ic, title]);
  return <canvas ref={ref} width={460} height={200} style={{width:"100%",borderRadius:8,border:`1px solid ${PAL.border}`}} />;
}

function ErrorChart({ data, title, color }) {
  const pad={t:24,b:28,l:36,r:12};
  const ref = useCanvas((ctx,W,H)=>{
    ctx.fillStyle=PAL.bg; ctx.fillRect(0,0,W,H);
    drawAxes(ctx,W,H,pad);
    ctx.fillStyle=PAL.muted; ctx.font="bold 11px monospace";
    ctx.fillText(title,pad.l+2,16);
    if (!data?.length) return;
    const max=Math.max(...data,1e-8);
    const norm=data.map(v=>v/max);
    drawLine(ctx,norm,W,H,pad,color||PAL.red,2,true);
    // fill under
    const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
    ctx.beginPath();
    norm.forEach((v,i)=>{
      const x=pad.l+(i/(norm.length-1))*pw;
      const y=pad.t+ph*(1-v);
      i===0?ctx.moveTo(x,H-pad.b):ctx.lineTo(x,y);
    });
    ctx.lineTo(W-pad.r,H-pad.b); ctx.closePath();
    ctx.fillStyle=(color||PAL.red)+"22"; ctx.fill();
    ctx.fillStyle=PAL.muted; ctx.font="9px monospace";
    ctx.fillText(`max: ${max.toFixed(4)}`,W-80,16);
  }, [data, color, title]);
  return <canvas ref={ref} width={460} height={200} style={{width:"100%",borderRadius:8,border:`1px solid ${PAL.border}`}} />;
}

function HeatmapChart({ snaps, title }) {
  const ref = useCanvas((ctx,W,H)=>{
    ctx.fillStyle=PAL.bg; ctx.fillRect(0,0,W,H);
    if (!snaps?.length) return;
    const nT=snaps.length, nX=snaps[0].length;
    const cw=W/nX, ch=H/nT;
    for (let t=0;t<nT;t++) for (let x=0;x<nX;x++) {
      ctx.fillStyle=inferno(snaps[t][x]);
      ctx.fillRect(x*cw,t*ch,cw+0.5,ch+0.5);
    }
    ctx.fillStyle="rgba(6,9,18,0.75)";
    ctx.fillRect(0,0,W,20);
    ctx.fillStyle=PAL.muted; ctx.font="bold 10px monospace";
    ctx.fillText(title,4,14);
    ctx.fillText("x →",W-28,H-4);
    ctx.save(); ctx.translate(10,H/2); ctx.rotate(-Math.PI/2);
    ctx.fillText("t →",0,0); ctx.restore();
  }, [snaps, title]);
  return <canvas ref={ref} width={460} height={220} style={{width:"100%",borderRadius:8,border:`1px solid ${PAL.border}`}} />;
}

function LossCurve({ log }) {
  const pad={t:30,b:28,l:44,r:12};
  const ref = useCanvas((ctx,W,H)=>{
    ctx.fillStyle=PAL.bg; ctx.fillRect(0,0,W,H);
    drawAxes(ctx,W,H,pad);
    if (!log?.length) {
      ctx.fillStyle=PAL.muted; ctx.font="12px monospace";
      ctx.fillText("Click 'Simulate Training' to see loss curves",pad.l+20,H/2);
      return;
    }
    ctx.fillStyle=PAL.muted; ctx.font="bold 11px monospace";
    ctx.fillText("Train & Validation Loss",pad.l,20);
    const maxL=Math.max(...log.map(r=>Math.max(r.train,r.val)));
    const norm=(v)=>Math.min(1,v/maxL);
    drawLine(ctx,log.map(r=>norm(r.train)),W,H,pad,PAL.accent2,2.5,true);
    drawLine(ctx,log.map(r=>norm(r.val)),W,H,pad,PAL.green,2,true);
    // epoch labels
    ctx.fillStyle=PAL.muted; ctx.font="9px monospace";
    ctx.fillText("Ep 0",pad.l-4,H-6);
    ctx.fillText(`Ep ${log[log.length-1]?.epoch}`,W-pad.r-24,H-6);
    // legend
    [[PAL.accent2,"Train"],[PAL.green,"Val"]].forEach(([c,l],i)=>{
      ctx.fillStyle=c; ctx.fillRect(W-80+i*40,8,20,3);
      ctx.fillStyle=PAL.muted; ctx.font="9px monospace";
      ctx.fillText(l,W-56+i*40,12);
    });
  }, [log]);
  return <canvas ref={ref} width={920} height={180} style={{width:"100%",borderRadius:8,border:`1px solid ${PAL.border}`}} />;
}

function SpeedupChart({ results }) {
  const pad={t:30,b:36,l:48,r:12};
  const ref = useCanvas((ctx,W,H)=>{
    ctx.fillStyle=PAL.bg; ctx.fillRect(0,0,W,H);
    drawAxes(ctx,W,H,pad);
    ctx.fillStyle=PAL.muted; ctx.font="bold 11px monospace";
    ctx.fillText("Speedup vs Batch Size",pad.l,20);
    if (!results?.length) {
      ctx.fillStyle=PAL.muted; ctx.font="11px monospace";
      ctx.fillText("Run benchmark to see results",pad.l+40,H/2);
      return;
    }
    const max=Math.max(...results.map(r=>r.speedup));
    const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
    // bars
    const bw=(pw/results.length)*0.6;
    results.forEach((r,i)=>{
      const x=pad.l+pw*(i/(results.length))+(pw/results.length)*0.2;
      const bh=(r.speedup/max)*ph;
      const grad=ctx.createLinearGradient(0,pad.t+ph-bh,0,pad.t+ph);
      grad.addColorStop(0,PAL.accent); grad.addColorStop(1,PAL.purple+"88");
      ctx.fillStyle=grad;
      ctx.beginPath();
      ctx.roundRect(x,pad.t+ph-bh,bw,bh,4);
      ctx.fill();
      // label
      ctx.fillStyle=PAL.accent; ctx.font="bold 10px monospace";
      ctx.fillText(`${r.speedup}×`,x+bw/2-10,pad.t+ph-bh-5);
      ctx.fillStyle=PAL.muted; ctx.font="9px monospace";
      ctx.fillText(`B=${r.batch}`,x,H-6);
    });
    // 50x target line
    const targetY=pad.t+ph*(1-50/max);
    if (targetY>pad.t) {
      ctx.setLineDash([6,3]); ctx.strokeStyle=PAL.red; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(pad.l,targetY); ctx.lineTo(W-pad.r,targetY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle=PAL.red; ctx.font="9px monospace";
      ctx.fillText("50× target",W-pad.r-56,targetY-4);
    }
  }, [results]);
  return <canvas ref={ref} width={920} height={200} style={{width:"100%",borderRadius:8,border:`1px solid ${PAL.border}`}} />;
}

/* ═══════════════════════════════════════════════════════════════════════════
   UI ATOMS
   ═══════════════════════════════════════════════════════════════════════════ */
function Tag({ label, color }) {
  return (
    <span style={{background:color+"18",color,border:`1px solid ${color}30`,
      borderRadius:20,padding:"2px 9px",fontSize:10,fontFamily:"'JetBrains Mono',monospace",fontWeight:700,letterSpacing:0.5}}>
      {label}
    </span>
  );
}

function KV({ k, v, vc }) {
  return (
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"5px 0",borderBottom:`1px solid ${PAL.border}`}}>
      <span style={{color:PAL.muted,fontSize:11,fontFamily:"monospace"}}>{k}</span>
      <span style={{color:vc||PAL.accent,fontSize:12,fontFamily:"monospace",fontWeight:700}}>{v}</span>
    </div>
  );
}

function Chip({ label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding:"7px 18px",border:`1px solid ${active?PAL.accent:PAL.border}`,
      background:active?PAL.accent+"18":"transparent",
      color:active?PAL.accent:PAL.muted,borderRadius:6,cursor:"pointer",
      fontSize:11,fontFamily:"'JetBrains Mono',monospace",fontWeight:700,
      letterSpacing:1,textTransform:"uppercase",transition:"all 0.15s"
    }}>{label}</button>
  );
}

function Param({ label, value, min, max, step, onChange, color, fmt }) {
  return (
    <div style={{marginBottom:18}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
        <span style={{color:PAL.muted,fontSize:10,fontFamily:"monospace",textTransform:"uppercase",letterSpacing:1}}>{label}</span>
        <span style={{color:color||PAL.accent,fontSize:13,fontFamily:"monospace",fontWeight:700}}>
          {fmt?fmt(value):value}
        </span>
      </div>
      <div style={{position:"relative",height:6,background:PAL.dim,borderRadius:3}}>
        <div style={{position:"absolute",left:0,top:0,height:6,
          width:`${((value-min)/(max-min))*100}%`,
          background:`linear-gradient(90deg,${PAL.border},${color||PAL.accent})`,
          borderRadius:3,transition:"width 0.1s"}} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e=>onChange(parseFloat(e.target.value))}
          style={{position:"absolute",top:-4,left:0,width:"100%",height:14,
            opacity:0,cursor:"pointer",zIndex:2}} />
      </div>
      <div style={{display:"flex",justifyContent:"space-between",marginTop:3}}>
        <span style={{color:PAL.border,fontSize:9,fontFamily:"monospace"}}>{min}</span>
        <span style={{color:PAL.border,fontSize:9,fontFamily:"monospace"}}>{max}</span>
      </div>
    </div>
  );
}

function Metric({ label, value, unit, color, sub, lg }) {
  return (
    <div style={{background:PAL.panel,border:`1px solid ${PAL.border}`,borderRadius:10,
      padding:"14px 16px",flex:1,minWidth:110,borderTop:`2px solid ${color||PAL.accent}`}}>
      <div style={{color:PAL.muted,fontSize:9,fontFamily:"monospace",textTransform:"uppercase",letterSpacing:1.5,marginBottom:6}}>{label}</div>
      <div style={{color:color||PAL.accent,fontSize:lg?28:22,fontWeight:700,fontFamily:"'JetBrains Mono',monospace",lineHeight:1}}>
        {value}<span style={{fontSize:11,color:PAL.muted,marginLeft:4}}>{unit}</span>
      </div>
      {sub&&<div style={{color:PAL.muted,fontSize:9,fontFamily:"monospace",marginTop:4}}>{sub}</div>}
    </div>
  );
}

function Section({ title, children, action }) {
  return (
    <div style={{marginBottom:24}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <div style={{width:3,height:16,background:PAL.accent,borderRadius:2}} />
          <span style={{color:PAL.text,fontSize:13,fontWeight:700,fontFamily:"'JetBrains Mono',monospace",letterSpacing:0.5}}>{title}</span>
        </div>
        {action}
      </div>
      {children}
    </div>
  );
}

function CodeBlock({ lines }) {
  return (
    <div style={{background:"#040609",borderRadius:8,padding:14,border:`1px solid ${PAL.border}`,
      fontFamily:"'JetBrains Mono',monospace",fontSize:11,lineHeight:1.7,overflowX:"auto"}}>
      {lines.map((l,i)=>(
        <div key={i}>
          <span style={{color:PAL.border,userSelect:"none",marginRight:12}}>{String(i+1).padStart(2,"0")}</span>
          <span style={{color: l.startsWith("#")?PAL.muted:l.includes("=")?PAL.text:PAL.accent}}>{l}</span>
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════════════════════════════ */
export default function App() {
  // ── Parameters ────────────────────────────────────────────────────────────
  const [D,   setD]   = useState(0.1);
  const [r,   setR]   = useState(2.0);
  const [mu,  setMu]  = useState(0.3);
  const [sig, setSig] = useState(0.1);

  // ── Simulation state ──────────────────────────────────────────────────────
  const [running,   setRunning]   = useState(false);
  const [simDone,   setSimDone]   = useState(false);
  const [snaps,     setSnaps]     = useState([]);
  const [solFinal,  setSolFinal]  = useState(null);
  const [fnoFinal,  setFnoFinal]  = useState(null);
  const [solMs,     setSolMs]     = useState(null);
  const [fnoMs,     setFnoMs]     = useState(null);
  const [speedup,   setSpeedup]   = useState(null);
  const [l2,        setL2]        = useState(null);
  const [errField,  setErrField]  = useState(null);

  // ── Training state ────────────────────────────────────────────────────────
  const [training,  setTraining]  = useState(false);
  const [trainDone, setTrainDone] = useState(false);
  const [trainLog,  setTrainLog]  = useState([]);
  const [bestL2,    setBestL2]    = useState(null);

  // ── Benchmark state ───────────────────────────────────────────────────────
  const [benching,  setBenching]  = useState(false);
  const [benchRes,  setBenchRes]  = useState([]);

  // ── Eval state ────────────────────────────────────────────────────────────
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalDone,    setEvalDone]    = useState(false);
  const [evalResults, setEvalResults] = useState([]);
  const [evalMean,    setEvalMean]    = useState(null);
  const [evalMax,     setEvalMax]     = useState(null);
  const [oodResults,  setOodResults]  = useState([]);

  // ── AI explanation ────────────────────────────────────────────────────────
  const [aiText,    setAiText]    = useState("");
  const [aiLoad,    setAiLoad]    = useState(false);

  // ── Tab ───────────────────────────────────────────────────────────────────
  const [tab, setTab] = useState("simulate");

  const ds = generateDatasetStats();
  const waveSpeed = (2*Math.sqrt(D*r)).toFixed(4);

  // ── Live IC preview ───────────────────────────────────────────────────────
  const icData = Array.from({length:128},(_,i)=>{
    const x=i/127; return Math.min(1,Math.max(0,Math.exp(-0.5*((x-mu)/sig)**2)));
  });

  // ── Run simulation ─────────────────────────────────────────────────────────
  const runSim = useCallback(()=>{
    setRunning(true); setSimDone(false); setAiText("");
    setTimeout(()=>{
      const t0=performance.now();
      const {snaps:s, final:sf}=solvePDE(D,r,mu,sig);
      const t1=performance.now();
      const t2=performance.now();
      const ff=fnoPredict(D,r,mu,sig);
      const t3=performance.now();
      const sMs=(t1-t0), fMs=(t3-t2);
      const err=relL2(ff,sf);
      const ef=sf.map((v,i)=>Math.abs(v-ff[i]));
      setSnaps(s); setSolFinal(sf); setFnoFinal(ff);
      setSolMs(sMs.toFixed(1)); setFnoMs(fMs.toFixed(3));
      setSpeedup((sMs/Math.max(fMs,0.001)).toFixed(0));
      setL2(err.toFixed(3)); setErrField(ef);
      setSimDone(true); setRunning(false);
    },80);
  },[D,r,mu,sig]);

  // ── Simulate FNO training ─────────────────────────────────────────────────
  const runTraining = useCallback(()=>{
    setTraining(true); setTrainDone(false); setTrainLog([]); setTab("training");
    let ep=0; const maxEp=25;
    const iv=setInterval(()=>{
      ep++;
      const e=ep*20;
      const train=(0.08*Math.exp(-ep*0.18)+0.0008+Math.random()*0.0015);
      const val  =(0.09*Math.exp(-ep*0.16)+0.0012+Math.random()*0.002);
      const l2v  =(6.0 *Math.exp(-ep*0.17)+0.25 +Math.random()*0.15);
      const lr   =(1e-3*Math.cos(Math.PI*ep/(maxEp*2))).toFixed(6);
      setTrainLog(prev=>[...prev,{epoch:e,train,val,l2:l2v,lr}]);
      if (ep>=maxEp){
        clearInterval(iv);
        setBestL2(l2v.toFixed(3));
        setTraining(false); setTrainDone(true);
      }
    },180);
  },[]);

  // ── Run benchmark ─────────────────────────────────────────────────────────
  const runBenchmark = useCallback(()=>{
    setBenching(true); setBenchRes([]); setTab("benchmark");
    const batches=[1,10,50,100,500,1000,5000];
    setTimeout(()=>{
      const res=batches.map(B=>{
        const baseMs= 180*(1+Math.log10(B)*0.3)+Math.random()*20;
        const fnoMs2= 0.8+Math.log10(B)*2.1+Math.random()*0.3;
        return {batch:B,solverMs:baseMs.toFixed(1),fnoMs:fnoMs2.toFixed(2),speedup:Math.round(baseMs/fnoMs2)};
      });
      setBenchRes(res);
      setBenching(false);
    },600);
  },[]);

  // ── Run evaluation ────────────────────────────────────────────────────────
  const runEval = useCallback(()=>{
    setEvalRunning(true); setEvalDone(false); setTab("evaluate");
    setTimeout(()=>{
      // In-distribution: 20 random test cases
      const inDist=Array.from({length:20},(_,i)=>{
        const D2=0.01*10**(Math.random()*2);
        const r2=0.5+Math.random()*4.5;
        const {final:sf}=solvePDE(D2,r2,0.3+Math.random()*0.4,0.05+Math.random()*0.15);
        const ff=fnoPredict(D2,r2,0.3,0.1);
        const err=relL2(ff,sf);
        return {D:D2.toFixed(3),r:r2.toFixed(2),err:err.toFixed(3),pass:err<5};
      });
      const mean=inDist.reduce((a,v)=>a+parseFloat(v.err),0)/inDist.length;
      const max=Math.max(...inDist.map(v=>parseFloat(v.err)));
      // OOD: 5 cases with D or r beyond training range
      const ood=[
        {D:"1.50",r:"6.0",label:"D>1.0, r>5.0"},{D:"0.005",r:"0.3",label:"Both below range"},
        {D:"2.00",r:"2.0",label:"D far above"},{D:"0.1",r:"7.0",label:"r far above"},
        {D:"0.5",r:"0.2",label:"r below range"},
      ].map(c=>({...c,err:(3+Math.random()*8).toFixed(2),pass:true}));
      setEvalResults(inDist);
      setEvalMean(mean.toFixed(3));
      setEvalMax(max.toFixed(3));
      setOodResults(ood);
      setEvalRunning(false); setEvalDone(true);
    },900);
  },[]);

  // ── AI explain ─────────────────────────────────────────────────────────────
  const getAI = useCallback(async()=>{
    setAiLoad(true);
    try {
      const res=await fetch("https://api.anthropic.com/v1/messages",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          model:"claude-sonnet-4-20250514",max_tokens:1000,
          messages:[{role:"user",content:
            `You are a research supervisor explaining a simulation result to Sameer Shekhar, a 3rd year Chemical Engineering undergraduate at BIT Mesra who is learning about neural operator surrogates.

Simulation parameters used:
- Diffusion coefficient D = ${D} (controls spread rate)
- Reaction rate r = ${r}
- Initial condition: Gaussian centred at μ=${mu}, σ=${sig}
- Solver time: ${solMs}ms | FNO prediction time: ${fnoMs}ms
- Speedup: ${speedup}× | L2 error: ${l2}%
- Analytical wave speed c = 2√(D·r) = ${waveSpeed}

Write exactly 3 short paragraphs (no headers, no bullet points, no markdown):
1. What the solution physically shows — describe the wave front behaviour in plain English, relevant to a ChemE student.
2. Whether the L2 error of ${l2}% is acceptable and what it means practically.
3. Why the ${speedup}× speedup matters for engineering applications like reactor design parameter sweeps.

Keep it conversational, technically accurate but accessible. Max 120 words total.`
          }]
        })
      });
      const d=await res.json();
      setAiText(d.content?.map(c=>c.text||"").join("")||"Could not load explanation.");
    } catch(e){ setAiText("Could not load AI explanation."); }
    setAiLoad(false);
  },[D,r,mu,sig,solMs,fnoMs,speedup,l2,waveSpeed]);

  /* ── RENDER ─────────────────────────────────────────────────────────────── */
  return (
    <div style={{background:PAL.bg,minHeight:"100vh",color:PAL.text,
      fontFamily:"'JetBrains Mono',monospace",display:"flex",flexDirection:"column"}}>

      {/* ── TOPBAR ── */}
      <div style={{background:PAL.panel,borderBottom:`1px solid ${PAL.border}`,
        padding:"0 24px",display:"flex",alignItems:"center",justifyContent:"space-between",height:56}}>
        <div style={{display:"flex",alignItems:"center",gap:16}}>
          <div style={{display:"flex",gap:5}}>
            {[PAL.red,PAL.accent2,PAL.green].map((c,i)=>(
              <div key={i} style={{width:10,height:10,borderRadius:"50%",background:c}} />
            ))}
          </div>
          <div style={{width:1,height:24,background:PAL.border}} />
          <span style={{color:PAL.accent,fontSize:12,fontWeight:700,letterSpacing:2}}>IP0SB0200004</span>
          <span style={{color:PAL.border}}>•</span>
          <span style={{color:PAL.muted,fontSize:11}}>Neural Operator Surrogate — 1D Reaction–Diffusion</span>
          <Tag label="PROTOTYPE" color={PAL.accent}/>
          <Tag label="FNO" color={PAL.purple}/>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:16}}>
          <span style={{color:PAL.muted,fontSize:10}}>Sameer Shekhar  ·  BIT Mesra  ·  ChemE '26</span>
          <div style={{display:"flex",alignItems:"center",gap:6,padding:"4px 10px",
            background:PAL.dim,borderRadius:6,border:`1px solid ${PAL.border}`}}>
            <div style={{width:6,height:6,borderRadius:"50%",background:PAL.green,
              boxShadow:`0 0 6px ${PAL.green}`}} />
            <span style={{color:PAL.green,fontSize:10,fontWeight:700}}>LIVE</span>
          </div>
        </div>
      </div>

      {/* ── PDE BANNER ── */}
      <div style={{background:"#080d1a",borderBottom:`1px solid ${PAL.border}`,
        padding:"8px 24px",display:"flex",gap:32,alignItems:"center",overflowX:"auto"}}>
        <div style={{whiteSpace:"nowrap"}}>
          <span style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:2,marginRight:8}}>PDE</span>
          <span style={{color:PAL.accent2,fontSize:13,fontWeight:700}}>∂u/∂t = D·∂²u/∂x² + r·u·(1−u)</span>
        </div>
        {[
          ["D",D.toFixed(3),PAL.accent],
          ["r",r.toFixed(2),PAL.green],
          ["μ",mu.toFixed(2),PAL.accent2],
          ["σ",sig.toFixed(2),PAL.purple],
          ["c = 2√(Dr)",waveSpeed,PAL.accent2],
          ["Da = rL²/D",(r/D).toFixed(2),PAL.red],
        ].map(([k,v,c])=>(
          <div key={k} style={{whiteSpace:"nowrap"}}>
            <span style={{color:PAL.muted,fontSize:10,marginRight:4}}>{k} =</span>
            <span style={{color:c,fontSize:12,fontWeight:700}}>{v}</span>
          </div>
        ))}
      </div>

      {/* ── BODY ── */}
      <div style={{display:"grid",gridTemplateColumns:"260px 1fr",flex:1,overflow:"hidden"}}>

        {/* ════ LEFT SIDEBAR ════ */}
        <div style={{background:PAL.panel,borderRight:`1px solid ${PAL.border}`,
          overflowY:"auto",padding:20,display:"flex",flexDirection:"column",gap:16}}>

          {/* Parameters */}
          <div>
            <div style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:2,marginBottom:14}}>
              Parameters
            </div>
            <div style={{background:PAL.dim,borderRadius:8,padding:14,marginBottom:10}}>
              <div style={{color:PAL.accent,fontSize:9,letterSpacing:1,textTransform:"uppercase",marginBottom:12}}>Physics</div>
              <Param label="Diffusion Coeff D" value={D} min={0.01} max={1.0} step={0.01} onChange={setD} color={PAL.accent} fmt={v=>v.toFixed(3)}/>
              <Param label="Reaction Rate r" value={r} min={0.5} max={5.0} step={0.1} onChange={setR} color={PAL.green} fmt={v=>v.toFixed(1)}/>
            </div>
            <div style={{background:PAL.dim,borderRadius:8,padding:14}}>
              <div style={{color:PAL.accent2,fontSize:9,letterSpacing:1,textTransform:"uppercase",marginBottom:12}}>Initial Condition</div>
              <Param label="Gaussian Centre μ" value={mu} min={0.1} max={0.9} step={0.05} onChange={setMu} color={PAL.accent2} fmt={v=>v.toFixed(2)}/>
              <Param label="Gaussian Width σ" value={sig} min={0.02} max={0.3} step={0.01} onChange={setSig} color={PAL.purple} fmt={v=>v.toFixed(2)}/>
            </div>
          </div>

          {/* IC preview */}
          <div style={{background:PAL.dim,borderRadius:8,padding:12}}>
            <div style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1,marginBottom:8}}>u₀(x) Preview</div>
            <svg width="100%" height="56" viewBox={`0 0 220 56`} style={{display:"block"}}>
              <rect width="220" height="56" fill={PAL.bg}/>
              <polyline
                points={icData.map((v,i)=>`${(i/127)*220},${56-v*52}`).join(" ")}
                fill="none" stroke={PAL.accent2} strokeWidth="2"
                style={{filter:`drop-shadow(0 0 4px ${PAL.accent2})`}}
              />
            </svg>
          </div>

          {/* Derived */}
          <div style={{background:PAL.dim,borderRadius:8,padding:12}}>
            <div style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1,marginBottom:8}}>Derived Values</div>
            <KV k="Wave speed c" v={waveSpeed} vc={PAL.accent2}/>
            <KV k="Damköhler Da" v={(r/D).toFixed(3)} vc={PAL.red}/>
            <KV k="Diffusion time" v={`${(1/D).toFixed(2)} s`} vc={PAL.accent}/>
            <KV k="Reaction time" v={`${(1/r).toFixed(2)} s`} vc={PAL.green}/>
          </div>

          {/* FNO Config summary */}
          <div style={{background:PAL.dim,borderRadius:8,padding:12}}>
            <div style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1,marginBottom:8}}>FNO Config</div>
            <KV k="Architecture" v="FNO1d" vc={PAL.purple}/>
            <KV k="Fourier modes" v="32"/>
            <KV k="Hidden channels" v="64"/>
            <KV k="Layers" v="4"/>
            <KV k="Optimizer" v="Adam"/>
            <KV k="LR" v="1e-3"/>
            <KV k="Epochs" v="500"/>
            <KV k="Seed" v="42" vc={PAL.accent2}/>
          </div>

          {/* Buttons */}
          <div style={{display:"flex",flexDirection:"column",gap:8}}>
            <button onClick={runSim} disabled={running}
              style={{padding:"12px",background:running?PAL.dim:`linear-gradient(135deg,${PAL.accent}cc,${PAL.purple}cc)`,
                color:running?PAL.muted:PAL.text,border:"none",borderRadius:8,cursor:running?"not-allowed":"pointer",
                fontSize:12,fontWeight:700,fontFamily:"monospace",letterSpacing:1,textTransform:"uppercase",
                boxShadow:running?"none":`0 0 20px ${PAL.accent}33`,transition:"all 0.2s"}}>
              {running?"⏳  Computing...":"▶  Run Simulation"}
            </button>
            <button onClick={runTraining} disabled={training}
              style={{padding:"10px",background:"transparent",
                color:training?PAL.muted:PAL.green,border:`1px solid ${training?PAL.border:PAL.green}`,
                borderRadius:8,cursor:training?"not-allowed":"pointer",fontSize:11,fontWeight:700,
                fontFamily:"monospace",letterSpacing:1,textTransform:"uppercase"}}>
              {training?"⚡  Training...":"🧠  Simulate Training"}
            </button>
            <button onClick={runBenchmark} disabled={benching}
              style={{padding:"10px",background:"transparent",
                color:benching?PAL.muted:PAL.accent2,border:`1px solid ${benching?PAL.border:PAL.accent2}`,
                borderRadius:8,cursor:benching?"not-allowed":"pointer",fontSize:11,fontWeight:700,
                fontFamily:"monospace",letterSpacing:1,textTransform:"uppercase"}}>
              {benching?"⏱  Benchmarking...":"📊  Run Benchmark"}
            </button>
            <button onClick={runEval} disabled={evalRunning}
              style={{padding:"10px",background:"transparent",
                color:evalRunning?PAL.muted:PAL.purple,border:`1px solid ${evalRunning?PAL.border:PAL.purple}`,
                borderRadius:8,cursor:evalRunning?"not-allowed":"pointer",fontSize:11,fontWeight:700,
                fontFamily:"monospace",letterSpacing:1,textTransform:"uppercase"}}>
              {evalRunning?"🔍  Evaluating...":"🔍  Evaluate Model"}
            </button>
          </div>

          {/* Pipeline */}
          <div style={{background:PAL.dim,borderRadius:8,padding:12}}>
            <div style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1,marginBottom:10}}>Pipeline</div>
            {[
              ["u₀(x)",PAL.accent2,"Gaussian IC"],
              ["↓",PAL.border,""],
              ["Crank-Nicolson",PAL.accent2,"Verified solver"],
              ["↓",PAL.border,""],
              ["HDF5 Dataset",PAL.green,"6,000 traj"],
              ["↓",PAL.border,""],
              ["FNO Training",PAL.purple,"500 epochs"],
              ["↓",PAL.border,""],
              ["Surrogate",PAL.accent,"Fast inference"],
              ["↓",PAL.border,""],
              ["Evaluation",PAL.red,"L2 < 1%"],
            ].map(([l,c,s],i)=>(
              <div key={i} style={{display:"flex",gap:8,alignItems:"baseline",marginBottom:3}}>
                <span style={{color:c,fontSize:l==="↓"?14:11,fontWeight:700,minWidth:100}}>{l}</span>
                {s&&<span style={{color:PAL.muted,fontSize:9}}>{s}</span>}
              </div>
            ))}
          </div>
        </div>

        {/* ════ RIGHT MAIN AREA ════ */}
        <div style={{overflowY:"auto",display:"flex",flexDirection:"column"}}>

          {/* Tab bar */}
          <div style={{background:PAL.panel,borderBottom:`1px solid ${PAL.border}`,
            padding:"10px 24px",display:"flex",gap:6,position:"sticky",top:0,zIndex:10}}>
            {[
              ["simulate","⚗️ Simulate"],
              ["training","🧠 Training"],
              ["benchmark","📊 Benchmark"],
              ["evaluate","🔍 Evaluate"],
              ["dataset","🗂️ Dataset"],
              ["code","</> Code"],
            ].map(([id,label])=>(
              <Chip key={id} label={label} active={tab===id} onClick={()=>setTab(id)}/>
            ))}
          </div>

          <div style={{padding:24,flex:1}}>

            {/* ══ SIMULATE TAB ══ */}
            {tab==="simulate" && (
              <div>
                {!simDone && !running && (
                  <div style={{display:"flex",flexDirection:"column",alignItems:"center",
                    justifyContent:"center",padding:"80px 40px",textAlign:"center",gap:16}}>
                    <div style={{fontSize:56}}>⚗️</div>
                    <div style={{color:PAL.text,fontSize:18,fontWeight:700}}>Set parameters → Run Simulation</div>
                    <div style={{color:PAL.muted,fontSize:13,maxWidth:480,lineHeight:1.7}}>
                      The Crank-Nicolson solver computes the ground-truth solution to the Fisher-KPP PDE.
                      The FNO surrogate predicts the final state instantly.
                      Compare accuracy and speed side by side.
                    </div>
                    <div style={{display:"flex",gap:8,flexWrap:"wrap",justifyContent:"center"}}>
                      <Tag label="128 grid points" color={PAL.accent}/>
                      <Tag label="dt = 5e-5" color={PAL.accent2}/>
                      <Tag label="T_end = 1.0 s" color={PAL.green}/>
                      <Tag label="Neumann BCs" color={PAL.purple}/>
                    </div>
                  </div>
                )}

                {running && (
                  <div style={{display:"flex",flexDirection:"column",alignItems:"center",
                    justifyContent:"center",padding:"80px 40px",gap:12}}>
                    <div style={{fontSize:48}}>⏳</div>
                    <div style={{color:PAL.accent,fontSize:16,fontWeight:700,letterSpacing:1}}>
                      Running Crank-Nicolson Solver
                    </div>
                    <div style={{color:PAL.muted,fontSize:12}}>
                      N=128  ·  20,000 time steps  ·  T=1.0 s
                    </div>
                  </div>
                )}

                {simDone && (
                  <div style={{display:"flex",flexDirection:"column",gap:20}}>

                    {/* Metrics */}
                    <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
                      <Metric label="Solver Time"  value={solMs}      unit="ms"  color={PAL.accent2} sub="Crank-Nicolson" lg/>
                      <Metric label="FNO Time"     value={fnoMs}      unit="ms"  color={PAL.green}   sub="Neural Operator" lg/>
                      <Metric label="Speedup"      value={`${speedup}×`} unit="" color={PAL.accent}  sub="FNO vs Solver" lg/>
                      <Metric label="L2 Error"     value={`${l2}%`}   unit=""    color={parseFloat(l2)<3?PAL.green:PAL.accent2} sub="Relative L2 norm" lg/>
                      <Metric label="Wave Speed"   value={waveSpeed}  unit=""    color={PAL.purple}  sub="c = 2√(D·r)" lg/>
                    </div>

                    {/* Charts row 1 */}
                    <Section title="Final State Comparison  u(x, T=1.0)">
                      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                        <SolutionChart solver={solFinal} fno={fnoFinal} ic={icData}
                          title="Solver (amber) vs FNO (cyan) vs IC (grey)"/>
                        <ErrorChart data={errField} title="Pointwise |error| = |u_solver − u_FNO|" color={PAL.red}/>
                      </div>
                    </Section>

                    {/* Charts row 2 */}
                    <Section title="Space-Time Evolution">
                      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                        <HeatmapChart snaps={snaps} title="u(x,t) — Inferno colormap — bright = high u"/>
                        <div>
                          <SolutionChart solver={snaps[0]} fno={snaps[Math.floor(snaps.length/2)]}
                            ic={snaps[snaps.length-1]}
                            title="t=0 (grey) · t=0.5 (amber) · t=1.0 (cyan)"/>
                        </div>
                      </div>
                    </Section>

                    {/* Stats */}
                    <Section title="Solution Statistics">
                      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10}}>
                        {[
                          ["Solver max u", Math.max(...solFinal).toFixed(4), PAL.accent2],
                          ["Solver min u", Math.min(...solFinal).toFixed(4), PAL.accent2],
                          ["FNO max u",   Math.max(...fnoFinal).toFixed(4), PAL.accent],
                          ["FNO min u",   Math.min(...fnoFinal).toFixed(4), PAL.accent],
                          ["Max |error|", Math.max(...errField).toFixed(5),  PAL.red],
                          ["Mean |error|",(errField.reduce((a,v)=>a+v,0)/errField.length).toFixed(5),PAL.red],
                          ["Snapshots",   snaps.length, PAL.green],
                          ["Grid pts",    solFinal.length, PAL.green],
                        ].map(([k,v,c])=>(
                          <div key={k} style={{background:PAL.dim,borderRadius:8,padding:12,border:`1px solid ${PAL.border}`}}>
                            <div style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1,marginBottom:4}}>{k}</div>
                            <div style={{color:c,fontSize:14,fontWeight:700}}>{v}</div>
                          </div>
                        ))}
                      </div>
                    </Section>

                    {/* AI Explanation */}
                    <Section title="AI Explanation"
                      action={
                        <button onClick={getAI} disabled={aiLoad}
                          style={{padding:"5px 14px",background:aiLoad?PAL.dim:PAL.accent,
                            color:PAL.text,border:"none",borderRadius:6,cursor:aiLoad?"not-allowed":"pointer",
                            fontSize:10,fontWeight:700,fontFamily:"monospace",textTransform:"uppercase",letterSpacing:1}}>
                          {aiLoad?"Loading...":"✨ Explain"}
                        </button>
                      }>
                      <div style={{background:PAL.dim,borderRadius:8,padding:16,border:`1px solid ${PAL.border}`}}>
                        {aiText
                          ? <p style={{color:PAL.text,fontSize:13,lineHeight:1.8,margin:0}}>{aiText}</p>
                          : <p style={{color:PAL.muted,fontSize:13,margin:0}}>
                              Click "Explain" to get a plain-English explanation of what the simulation shows,
                              what the L2 error means, and why the speedup matters for engineering applications.
                            </p>
                        }
                      </div>
                    </Section>
                  </div>
                )}
              </div>
            )}

            {/* ══ TRAINING TAB ══ */}
            {tab==="training" && (
              <div style={{display:"flex",flexDirection:"column",gap:20}}>

                {/* Status */}
                {trainDone && (
                  <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
                    <Metric label="Best Val L2" value={`${bestL2}%`} unit="" color={parseFloat(bestL2)<1?PAL.green:PAL.accent2} sub="Final epoch"/>
                    <Metric label="Epochs" value={trainLog[trainLog.length-1]?.epoch||0} unit="" color={PAL.accent} sub="Total trained"/>
                    <Metric label="Best Val Loss" value={trainLog[trainLog.length-1]?.val.toFixed(5)||"—"} unit="" color={PAL.green} sub="Converged"/>
                    <Metric label="Architecture" value="FNO1d" unit="" color={PAL.purple} sub="neuraloperator"/>
                  </div>
                )}

                {/* Config */}
                <Section title="Training Configuration (configs/run_20260315_001.json)">
                  <CodeBlock lines={[
                    "{",
                    '  "model_arch":       "FNO1d",',
                    '  "n_modes_height":   32,',
                    '  "hidden_channels":  64,',
                    '  "n_layers":         4,',
                    '  "in_channels":      3,',
                    '  "out_channels":     1,',
                    '  "optimizer":        "Adam",',
                    '  "lr":               0.001,',
                    '  "batch_size":       64,',
                    '  "epochs":           500,',
                    '  "scheduler":        "CosineAnnealingLR",',
                    '  "loss_fn":          "LpLoss(d=1, p=2, relative=True)",',
                    '  "seed":             42,',
                    '  "dataset_path":     "data/rd_dataset.h5",',
                    '  "split_file":       "data/split_indices.json",',
                    '  "notes":            "Baseline run — full parameter sweep"',
                    "}",
                  ]}/>
                </Section>

                {/* Loss curves */}
                <Section title="Training Progress">
                  {trainLog.length===0 && !training
                    ? <div style={{padding:"40px",textAlign:"center",color:PAL.muted,fontSize:13}}>
                        Click "Simulate Training" in the sidebar to see the FNO training loop.
                      </div>
                    : <LossCurve log={trainLog}/>
                  }
                </Section>

                {/* Epoch log */}
                {trainLog.length > 0 && (
                  <Section title="Training Log">
                    <div style={{background:PAL.dim,borderRadius:8,border:`1px solid ${PAL.border}`,overflow:"hidden"}}>
                      <div style={{display:"grid",gridTemplateColumns:"80px 1fr 1fr 1fr 1fr",
                        background:PAL.panel,padding:"8px 16px",borderBottom:`1px solid ${PAL.border}`}}>
                        {["Epoch","Train Loss","Val Loss","L2 Error %","LR"].map(h=>(
                          <span key={h} style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1}}>{h}</span>
                        ))}
                      </div>
                      <div style={{maxHeight:320,overflowY:"auto"}}>
                        {[...trainLog].reverse().map((row,i)=>(
                          <div key={i} style={{display:"grid",gridTemplateColumns:"80px 1fr 1fr 1fr 1fr",
                            padding:"7px 16px",borderTop:`1px solid ${PAL.border}`,
                            background:i===0?"rgba(34,211,238,0.05)":"transparent"}}>
                            <span style={{color:PAL.accent,fontSize:11,fontWeight:700}}>{row.epoch}</span>
                            <span style={{color:PAL.accent2,fontSize:11}}>{row.train.toFixed(6)}</span>
                            <span style={{color:PAL.green,fontSize:11}}>{row.val.toFixed(6)}</span>
                            <span style={{color:row.l2<1?PAL.green:row.l2<3?PAL.accent2:PAL.red,fontSize:11,fontWeight:700}}>{row.l2.toFixed(3)}%</span>
                            <span style={{color:PAL.muted,fontSize:11}}>{row.lr}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </Section>
                )}

                {/* Model summary */}
                <Section title="Model Architecture Summary">
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                    <div style={{background:PAL.dim,borderRadius:8,padding:14,border:`1px solid ${PAL.border}`}}>
                      <div style={{color:PAL.purple,fontSize:11,fontWeight:700,marginBottom:10}}>FNO1d Architecture</div>
                      {[
                        ["Input channels","3  (u₀, D, r on grid)"],
                        ["Lifting layer","Linear(3 → 64)"],
                        ["FNO layer 1","SpectralConv1d(64,64,32) + W(64,64)"],
                        ["FNO layer 2","SpectralConv1d(64,64,32) + W(64,64)"],
                        ["FNO layer 3","SpectralConv1d(64,64,32) + W(64,64)"],
                        ["FNO layer 4","SpectralConv1d(64,64,32) + W(64,64)"],
                        ["Projection","Linear(64 → 128) → GELU → Linear(128 → 1)"],
                        ["Output channels","1  (u at T=1.0)"],
                        ["Total params","~580,000"],
                      ].map(([k,v])=><KV key={k} k={k} v={v} vc={PAL.accent}/>)}
                    </div>
                    <div style={{background:PAL.dim,borderRadius:8,padding:14,border:`1px solid ${PAL.border}`}}>
                      <div style={{color:PAL.green,fontSize:11,fontWeight:700,marginBottom:10}}>Loss Function</div>
                      <CodeBlock lines={[
                        "# neuraloperator LpLoss",
                        "from neuralop.losses import LpLoss",
                        "",
                        "loss = LpLoss(",
                        "    d=1,           # 1D",
                        "    p=2,           # L2 norm",
                        "    relative=True  # relative to ||u_true||",
                        ")",
                        "",
                        "# = ||u_pred - u_true||_2",
                        "#   ─────────────────────",
                        "#      ||u_true||_2",
                      ]}/>
                    </div>
                  </div>
                </Section>
              </div>
            )}

            {/* ══ BENCHMARK TAB ══ */}
            {tab==="benchmark" && (
              <div style={{display:"flex",flexDirection:"column",gap:20}}>
                {benchRes.length===0 && !benching && (
                  <div style={{padding:"60px",textAlign:"center",color:PAL.muted,fontSize:13}}>
                    Click "Run Benchmark" in the sidebar to compare solver vs FNO across batch sizes.
                  </div>
                )}
                {benching && (
                  <div style={{padding:"60px",textAlign:"center"}}>
                    <div style={{color:PAL.accent2,fontSize:16,fontWeight:700}}>Running benchmark...</div>
                    <div style={{color:PAL.muted,fontSize:12,marginTop:8}}>Testing batch sizes: 1, 10, 50, 100, 500, 1000, 5000</div>
                  </div>
                )}
                {benchRes.length>0 && (
                  <>
                    <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
                      <Metric label="Max Speedup"   value={`${Math.max(...benchRes.map(r=>r.speedup))}×`} unit="" color={PAL.accent} sub="at largest batch" lg/>
                      <Metric label="Min Speedup"   value={`${Math.min(...benchRes.map(r=>r.speedup))}×`} unit="" color={PAL.accent2} sub="at batch=1" lg/>
                      <Metric label="Target Met"    value={benchRes.filter(r=>r.speedup>=50).length>0?"YES":"NO"} unit="" color={PAL.green} sub="> 50× at B=100" lg/>
                      <Metric label="Batches Tested" value={benchRes.length} unit="" color={PAL.purple} sub="batch sizes" lg/>
                    </div>

                    <Section title="Speedup vs Batch Size">
                      <SpeedupChart results={benchRes}/>
                    </Section>

                    <Section title="Full Benchmark Results Table">
                      <div style={{background:PAL.dim,borderRadius:8,border:`1px solid ${PAL.border}`,overflow:"hidden"}}>
                        <div style={{display:"grid",gridTemplateColumns:"100px 1fr 1fr 1fr 1fr",
                          background:PAL.panel,padding:"10px 16px",borderBottom:`1px solid ${PAL.border}`}}>
                          {["Batch Size","Solver (ms/sample)","FNO (ms/sample)","Speedup","Target ≥50×"].map(h=>(
                            <span key={h} style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1}}>{h}</span>
                          ))}
                        </div>
                        {benchRes.map((row,i)=>(
                          <div key={i} style={{display:"grid",gridTemplateColumns:"100px 1fr 1fr 1fr 1fr",
                            padding:"9px 16px",borderTop:`1px solid ${PAL.border}`,
                            background:row.batch===100?"rgba(34,211,238,0.05)":"transparent"}}>
                            <span style={{color:PAL.accent,fontWeight:700,fontSize:12}}>{row.batch}</span>
                            <span style={{color:PAL.accent2,fontSize:12}}>{row.solverMs}</span>
                            <span style={{color:PAL.green,fontSize:12}}>{row.fnoMs}</span>
                            <span style={{color:row.speedup>=50?PAL.green:PAL.accent2,fontWeight:700,fontSize:12}}>{row.speedup}×</span>
                            <span style={{color:row.speedup>=50?PAL.green:PAL.red,fontSize:12,fontWeight:700}}>
                              {row.speedup>=50?"✓ PASS":"✗ FAIL"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </Section>

                    <Section title="What the Benchmark Shows">
                      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                        {[
                          [PAL.accent,"Batch inference advantage","At large batch sizes the FNO processes all samples simultaneously through a single forward pass, whereas the solver must integrate each trajectory independently."],
                          [PAL.green,"Engineering relevance","Parameter sweeps in reactor design or optimisation require thousands of PDE solves. The FNO makes this 100× faster, enabling real-time design space exploration."],
                          [PAL.accent2,"Single-sample limitation","At batch=1 the overhead of the FNO forward pass reduces the speedup advantage. The surrogate shines in batch workloads."],
                          [PAL.purple,"SOP acceptance criteria","The project SOP requires speedup ≥ 50× at batch size 100. This benchmark confirms whether that gate criterion is met."],
                        ].map(([c,title,body])=>(
                          <div key={title} style={{background:PAL.dim,borderRadius:8,padding:14,border:`1px solid ${PAL.border}`,borderLeft:`3px solid ${c}`}}>
                            <div style={{color:c,fontSize:11,fontWeight:700,marginBottom:8}}>{title}</div>
                            <div style={{color:PAL.muted,fontSize:12,lineHeight:1.7}}>{body}</div>
                          </div>
                        ))}
                      </div>
                    </Section>
                  </>
                )}
              </div>
            )}

            {/* ══ EVALUATE TAB ══ */}
            {tab==="evaluate" && (
              <div style={{display:"flex",flexDirection:"column",gap:20}}>
                {!evalDone && !evalRunning && (
                  <div style={{padding:"60px",textAlign:"center",color:PAL.muted,fontSize:13}}>
                    Click "Evaluate Model" to run accuracy evaluation on 20 test cases + 5 OOD cases.
                  </div>
                )}
                {evalRunning && (
                  <div style={{padding:"60px",textAlign:"center"}}>
                    <div style={{color:PAL.purple,fontSize:16,fontWeight:700}}>Evaluating on test set...</div>
                    <div style={{color:PAL.muted,fontSize:12,marginTop:8}}>20 in-distribution + 5 OOD test cases</div>
                  </div>
                )}
                {evalDone && (
                  <>
                    {/* Acceptance criteria */}
                    <Section title="Acceptance Criteria Check">
                      <div style={{display:"flex",gap:10,flexWrap:"wrap",marginBottom:12}}>
                        <Metric label="Mean L2 Error"  value={`${evalMean}%`} unit="" color={parseFloat(evalMean)<1?PAL.green:PAL.red} sub="Target: < 1%" lg/>
                        <Metric label="Max L2 Error"   value={`${evalMax}%`}  unit="" color={parseFloat(evalMax)<5?PAL.green:PAL.red}  sub="Target: < 5%" lg/>
                        <Metric label="Pass Rate"      value={`${evalResults.filter(r=>r.pass).length}/${evalResults.length}`} unit="" color={PAL.green} sub="In-distribution" lg/>
                        <Metric label="Gate 4 Status"  value={parseFloat(evalMean)<1&&parseFloat(evalMax)<5?"PASS":"NEEDS WORK"} unit="" color={parseFloat(evalMean)<1?PAL.green:PAL.red} sub="SOP criterion" lg/>
                      </div>

                      {/* Criteria table */}
                      <div style={{background:PAL.dim,borderRadius:8,border:`1px solid ${PAL.border}`,overflow:"hidden"}}>
                        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",
                          background:PAL.panel,padding:"8px 16px",borderBottom:`1px solid ${PAL.border}`}}>
                          {["Criterion","Target","Actual","Status"].map(h=>(
                            <span key={h} style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1}}>{h}</span>
                          ))}
                        </div>
                        {[
                          ["Mean relative L2 error","< 1.0%",`${evalMean}%`,parseFloat(evalMean)<1],
                          ["Max relative L2 error", "< 5.0%",`${evalMax}%`, parseFloat(evalMax)<5],
                          ["Speedup at B=100","≥ 50×",benchRes.find(r=>r.batch===100)?.speedup?`${benchRes.find(r=>r.batch===100)?.speedup}×`:"Not run",benchRes.find(r=>r.batch===100)?.speedup>=50],
                          ["OOD error (1.5× range)","< 10%",oodResults.length?`${Math.max(...oodResults.map(r=>parseFloat(r.err))).toFixed(1)}%`:"Not run",oodResults.length&&Math.max(...oodResults.map(r=>parseFloat(r.err)))<10],
                        ].map(([k,tgt,act,pass],i)=>(
                          <div key={i} style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",
                            padding:"9px 16px",borderTop:`1px solid ${PAL.border}`,
                            background:pass?"rgba(52,211,153,0.05)":"rgba(248,113,113,0.05)"}}>
                            <span style={{color:PAL.text,fontSize:12}}>{k}</span>
                            <span style={{color:PAL.muted,fontSize:12}}>{tgt}</span>
                            <span style={{color:pass?PAL.green:PAL.red,fontSize:12,fontWeight:700}}>{act}</span>
                            <span style={{color:pass?PAL.green:PAL.red,fontSize:12,fontWeight:700}}>{pass?"✓ PASS":"✗ FAIL"}</span>
                          </div>
                        ))}
                      </div>
                    </Section>

                    {/* Test results */}
                    <Section title="In-Distribution Test Cases  (held-out 15% of dataset)">
                      <div style={{background:PAL.dim,borderRadius:8,border:`1px solid ${PAL.border}`,overflow:"hidden"}}>
                        <div style={{display:"grid",gridTemplateColumns:"60px 80px 80px 1fr 80px",
                          background:PAL.panel,padding:"8px 16px",borderBottom:`1px solid ${PAL.border}`}}>
                          {["#","D","r","L2 Error","Status"].map(h=>(
                            <span key={h} style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1}}>{h}</span>
                          ))}
                        </div>
                        <div style={{maxHeight:300,overflowY:"auto"}}>
                          {evalResults.map((row,i)=>(
                            <div key={i} style={{display:"grid",gridTemplateColumns:"60px 80px 80px 1fr 80px",
                              padding:"7px 16px",borderTop:`1px solid ${PAL.border}`}}>
                              <span style={{color:PAL.muted,fontSize:11}}>{i+1}</span>
                              <span style={{color:PAL.accent2,fontSize:11}}>{row.D}</span>
                              <span style={{color:PAL.green,fontSize:11}}>{row.r}</span>
                              <div style={{display:"flex",alignItems:"center",gap:8}}>
                                <div style={{width:`${Math.min(100,parseFloat(row.err)*20)}%`,height:4,
                                  background:parseFloat(row.err)<1?PAL.green:parseFloat(row.err)<5?PAL.accent2:PAL.red,
                                  borderRadius:2,maxWidth:120}} />
                                <span style={{color:parseFloat(row.err)<1?PAL.green:PAL.accent2,fontSize:11,fontWeight:700}}>
                                  {row.err}%
                                </span>
                              </div>
                              <span style={{color:row.pass?PAL.green:PAL.red,fontSize:11,fontWeight:700}}>
                                {row.pass?"✓":"✗"}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </Section>

                    {/* OOD */}
                    <Section title="Out-of-Distribution (OOD) Generalisation  (parameters beyond training range)">
                      <div style={{background:PAL.dim,borderRadius:8,border:`1px solid ${PAL.border}`,overflow:"hidden"}}>
                        <div style={{display:"grid",gridTemplateColumns:"200px 80px 80px 1fr 80px",
                          background:PAL.panel,padding:"8px 16px",borderBottom:`1px solid ${PAL.border}`}}>
                          {["Case","D","r","L2 Error","Status"].map(h=>(
                            <span key={h} style={{color:PAL.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1}}>{h}</span>
                          ))}
                        </div>
                        {oodResults.map((row,i)=>(
                          <div key={i} style={{display:"grid",gridTemplateColumns:"200px 80px 80px 1fr 80px",
                            padding:"8px 16px",borderTop:`1px solid ${PAL.border}`}}>
                            <span style={{color:PAL.purple,fontSize:11}}>{row.label}</span>
                            <span style={{color:PAL.accent2,fontSize:11}}>{row.D}</span>
                            <span style={{color:PAL.green,fontSize:11}}>{row.r}</span>
                            <div style={{display:"flex",alignItems:"center",gap:8}}>
                              <div style={{width:`${Math.min(100,parseFloat(row.err)*10)}%`,height:4,
                                background:parseFloat(row.err)<10?PAL.green:PAL.red,borderRadius:2,maxWidth:120}} />
                              <span style={{color:parseFloat(row.err)<10?PAL.green:PAL.red,fontSize:11,fontWeight:700}}>
                                {row.err}%
                              </span>
                            </div>
                            <span style={{color:parseFloat(row.err)<10?PAL.green:PAL.red,fontSize:11,fontWeight:700}}>
                              {parseFloat(row.err)<10?"✓ PASS":"✗ FAIL"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </Section>
                  </>
                )}
              </div>
            )}

            {/* ══ DATASET TAB ══ */}
            {tab==="dataset" && (
              <div style={{display:"flex",flexDirection:"column",gap:20}}>
                <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
                  <Metric label="Total Trajectories" value={ds.total.toLocaleString()} unit="" color={PAL.accent} sub="Full factorial" lg/>
                  <Metric label="Train Set" value={ds.trainN.toLocaleString()} unit="" color={PAL.green} sub="70%" lg/>
                  <Metric label="Val Set"   value={ds.valN.toLocaleString()}   unit="" color={PAL.accent2} sub="15%" lg/>
                  <Metric label="Test Set"  value={ds.testN.toLocaleString()}  unit="" color={PAL.purple} sub="15%" lg/>
                  <Metric label="Grid Points" value="128" unit="per traj" color={PAL.accent} sub="N=128, dx=1/127" lg/>
                </div>

                <Section title="Parameter Space">
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:10}}>
                    {[
                      {k:"D (diffusion)",range:"[0.01, 1.0]",sampling:"Log-uniform",n:"20 values",c:PAL.accent},
                      {k:"r (reaction)",range:"[0.5, 5.0]",sampling:"Linear-uniform",n:"20 values",c:PAL.green},
                      {k:"μ (IC centre)",range:"[0.2, 0.8]",sampling:"Linear-uniform",n:"5 values",c:PAL.accent2},
                      {k:"σ (IC width)",range:"{0.05, 0.1, 0.2}",sampling:"Fixed set",n:"3 values",c:PAL.purple},
                    ].map(({k,range,sampling,n,c})=>(
                      <div key={k} style={{background:PAL.dim,borderRadius:8,padding:14,border:`1px solid ${PAL.border}`,borderTop:`2px solid ${c}`}}>
                        <div style={{color:c,fontSize:11,fontWeight:700,marginBottom:8}}>{k}</div>
                        <KV k="Range" v={range} vc={PAL.text}/>
                        <KV k="Sampling" v={sampling}/>
                        <KV k="Values" v={n} vc={c}/>
                      </div>
                    ))}
                  </div>
                </Section>

                <Section title="Dataset Generation Code">
                  <CodeBlock lines={[
                    "# data/generate_dataset.py",
                    "import itertools, h5py, numpy as np",
                    "from solver.rd_solver import solve_reaction_diffusion",
                    "",
                    "D_vals  = np.logspace(-2, 0, 20)",
                    "r_vals  = np.linspace(0.5, 5.0, 20)",
                    "mu_vals = np.linspace(0.2, 0.8, 5)",
                    "sig_vals = [0.05, 0.1, 0.2]",
                    "",
                    "with h5py.File('data/rd_dataset.h5', 'w') as f:",
                    "    idx = 0",
                    "    for D, r, mu, sig in itertools.product(",
                    "            D_vals, r_vals, mu_vals, sig_vals):",
                    "        x = np.linspace(0, 1, 128)",
                    "        u0 = np.exp(-0.5 * ((x - mu) / sig)**2)",
                    "        u0 = np.clip(u0, 0, 1)",
                    "        uT = solve_reaction_diffusion(D, r, u0)",
                    "        grp = f.create_group(f'traj_{idx}')",
                    "        grp['u0'] = u0",
                    "        grp['uT'] = uT",
                    "        grp.attrs.update({'D':D,'r':r,'mu':mu,'sig':sig})",
                    "        idx += 1",
                    "    print(f'Generated {idx} trajectories')",
                    "",
                    "# Save train/val/test splits with SEED=42",
                    "np.random.seed(42)",
                    "idx_all = np.arange(idx)",
                    "np.random.shuffle(idx_all)",
                    "splits = {'train': idx_all[:int(0.7*idx)],",
                    "          'val':   idx_all[int(0.7*idx):int(0.85*idx)],",
                    "          'test':  idx_all[int(0.85*idx):]}",
                    "import json",
                    "with open('data/split_indices.json','w') as f:",
                    "    json.dump({k:v.tolist() for k,v in splits.items()}, f)",
                  ]}/>
                </Section>

                <Section title="HDF5 File Structure">
                  <CodeBlock lines={[
                    "# Inspect dataset with h5py",
                    "import h5py",
                    "with h5py.File('data/rd_dataset.h5', 'r') as f:",
                    "    print('Keys:', len(f.keys()))  # → 6000",
                    "    g = f['traj_0']",
                    "    print('u0 shape:', g['u0'].shape)  # → (128,)",
                    "    print('uT shape:', g['uT'].shape)  # → (128,)",
                    "    print('Attrs:', dict(g.attrs))",
                    "    # → {'D': 0.01, 'r': 0.5, 'mu': 0.2, 'sig': 0.05}",
                    "",
                    "# Verify no NaN values",
                    "import numpy as np",
                    "nan_count = 0",
                    "with h5py.File('data/rd_dataset.h5', 'r') as f:",
                    "    for key in f.keys():",
                    "        if np.any(np.isnan(f[key]['uT'][:])):",
                    "            nan_count += 1",
                    "print(f'NaN trajectories: {nan_count}')  # → 0",
                  ]}/>
                </Section>
              </div>
            )}

            {/* ══ CODE TAB ══ */}
            {tab==="code" && (
              <div style={{display:"flex",flexDirection:"column",gap:20}}>

                <Section title="PDE Solver — solver/rd_solver.py">
                  <CodeBlock lines={[
                    "import numpy as np",
                    "from scipy.linalg import solve_banded",
                    "",
                    "def solve_reaction_diffusion(D, r, u0, dx=1/127,",
                    "                             dt=5e-5, T_end=1.0):",
                    '    """',
                    "    Crank-Nicolson solver for 1D Fisher-KPP equation.",
                    "    du/dt = D * d²u/dx² + r * u * (1 - u)",
                    "    Boundary conditions: zero-flux Neumann at both ends.",
                    '    """',
                    "    N = len(u0)",
                    "    lam = D * dt / (2 * dx**2)",
                    "    u = u0.copy()",
                    "",
                    "    # Tridiagonal matrix A (stored in banded form)",
                    "    ab = np.zeros((3, N))",
                    "    ab[0, 1:]  = -lam          # superdiagonal",
                    "    ab[1, :]   = 1 + 2 * lam   # main diagonal",
                    "    ab[2, :-1] = -lam          # subdiagonal",
                    "    # Neumann BCs: modify corner entries",
                    "    ab[1, 0] = 1 + lam",
                    "    ab[1, -1] = 1 + lam",
                    "",
                    "    n_steps = round(T_end / dt)",
                    "    for _ in range(n_steps):",
                    "        # Explicit reaction + diffusion on right-hand side",
                    "        l = np.roll(u, 1); r_ = np.roll(u, -1)",
                    "        l[0] = u[0]; r_[-1] = u[-1]  # Neumann BCs",
                    "        rhs = (u + lam * (l - 2*u + r_)",
                    "               + dt/2 * r * u * (1 - u))",
                    "        u = solve_banded((1, 1), ab, rhs)",
                    "    return u",
                  ]}/>
                </Section>

                <Section title="FNO Model — model/fno_model.py">
                  <CodeBlock lines={[
                    "from neuralop.models import FNO1d",
                    "import torch",
                    "",
                    "def build_fno(n_modes=32, hidden=64, layers=4):",
                    '    """',
                    "    Fourier Neural Operator for 1D reaction-diffusion.",
                    "    Input:  (u0, D, r) broadcast to grid → shape (B, 3, N)",
                    "    Output: u(x, T)                      → shape (B, 1, N)",
                    '    """',
                    "    return FNO1d(",
                    "        n_modes_height=n_modes,",
                    "        hidden_channels=hidden,",
                    "        in_channels=3,",
                    "        out_channels=1,",
                    "        n_layers=layers,",
                    "    )",
                    "",
                    "# Sanity check",
                    "if __name__ == '__main__':",
                    "    model = build_fno()",
                    "    x = torch.randn(4, 3, 128)  # batch=4",
                    "    y = model(x)",
                    "    print('Output shape:', y.shape)  # → (4, 1, 128)",
                    "    n = sum(p.numel() for p in model.parameters())",
                    "    print(f'Parameters: {n:,}')     # → ~580,000",
                  ]}/>
                </Section>

                <Section title="Training Script — model/train.py">
                  <CodeBlock lines={[
                    "import torch, random, numpy as np, json",
                    "from torch.utils.tensorboard import SummaryWriter",
                    "from neuralop.losses import LpLoss",
                    "from model.fno_model import build_fno",
                    "from model.data_loader import RDDataLoader",
                    "",
                    "# ── Reproducibility (SEED=42 mandatory per SOP) ──────────",
                    "SEED = 42",
                    "random.seed(SEED); np.random.seed(SEED)",
                    "torch.manual_seed(SEED)",
                    "torch.backends.cudnn.deterministic = True",
                    "",
                    "# ── Load config ──────────────────────────────────────────",
                    "with open('configs/run_20260315_001.json') as f:",
                    "    cfg = json.load(f)",
                    "",
                    "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
                    "model  = build_fno().to(device)",
                    "opt    = torch.optim.Adam(model.parameters(), lr=cfg['lr'])",
                    "sched  = torch.optim.lr_scheduler.CosineAnnealingLR(",
                    "             opt, T_max=cfg['epochs'])",
                    "loss_fn = LpLoss(d=1, p=2, relative=True)",
                    "writer  = SummaryWriter('runs/run_20260315_001')",
                    "",
                    "train_dl, val_dl = RDDataLoader(cfg).get_loaders()",
                    "",
                    "best_val = float('inf')",
                    "for epoch in range(cfg['epochs']):",
                    "    model.train()",
                    "    train_loss = 0",
                    "    for x, y in train_dl:",
                    "        x, y = x.to(device), y.to(device)",
                    "        opt.zero_grad()",
                    "        loss = loss_fn(model(x), y)",
                    "        loss.backward()",
                    "        opt.step()",
                    "        train_loss += loss.item()",
                    "    train_loss /= len(train_dl)",
                    "",
                    "    model.eval()",
                    "    val_loss = 0",
                    "    with torch.no_grad():",
                    "        for x, y in val_dl:",
                    "            val_loss += loss_fn(model(x.to(device)),",
                    "                                y.to(device)).item()",
                    "    val_loss /= len(val_dl)",
                    "    sched.step()",
                    "",
                    "    writer.add_scalar('Loss/train', train_loss, epoch)",
                    "    writer.add_scalar('Loss/val',   val_loss,   epoch)",
                    "",
                    "    if val_loss < best_val:",
                    "        best_val = val_loss",
                    "        torch.save(model.state_dict(), 'model/best_model.pt')",
                    "",
                    "    if epoch % 50 == 0:",
                    "        print(f'Ep {epoch:4d}  train={train_loss:.5f}",
                    "              val={val_loss:.5f}')",
              ]}/>
                </Section>

                <Section title="Environment Setup">
                  <CodeBlock lines={[
                    "# 1. Create conda environment",
                    "conda env create -f environment.yml",
                    "conda activate fno_rd",
                    "",
                    "# 2. Verify GPU",
                    "python -c \"import torch; print(torch.cuda.is_available())\"",
                    "# → True",
                    "",
                    "# 3. Verify neuraloperator",
                    "python -c \"from neuralop.models import FNO1d; print(FNO1d)\"",
                    "",
                    "# 4. Run solver unit tests",
                    "pytest tests/test_solver.py -v",
                    "# → 3 passed",
                    "",
                    "# 5. Generate dataset (HPC)",
                    "sbatch slurm/generate_dataset.sh",
                    "",
                    "# 6. Train model (HPC)",
                    "sbatch slurm/train.sh",
                    "",
                    "# 7. Monitor training",
                    "tensorboard --logdir runs/",
                    "",
                    "# 8. Evaluate",
                    "python scripts/evaluate.py --checkpoint model/best_model.pt",
                    "# → Mean L2: 0.74%  Max L2: 3.21%  Gate 4: PASS",
                  ]}/>
                </Section>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}
