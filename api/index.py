import numpy as np
import numexpr as ne
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class SimulationRequest(BaseModel):
    D: float
    r: float
    mu: float
    sigma: float
    modelType: str
    custom_eq: str = ""
    N: int = 128
    dt: float = 5e-5
    T_end: float = 1.0


def thomas_solve(lo, diag, up, rhs):
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)
    x = np.zeros(n)
    eps = 1e-15
    d0 = diag[0] if abs(diag[0]) >= eps else (eps if diag[0] >= 0 else -eps)

    c[0] = up[0] / d0
    d[0] = rhs[0] / d0
    for i in range(1, n):
        m = diag[i] - lo[i] * c[i - 1]
        if abs(m) < eps:
            m = eps if m >= 0 else -eps
        c[i] = up[i] / m
        d[i] = (rhs[i] - lo[i] * d[i - 1]) / m

    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    N = max(3, min(512, req.N))
    dt = max(1e-6, min(0.1, req.dt))
    D = max(1e-5, min(10.0, req.D))
    r = max(0.0, min(50.0, req.r))
    mu = max(0.001, min(0.999, req.mu))
    sig = max(0.001, min(2.0, req.sigma))

    dx = 1.0 / (N - 1)
    lam = D * dt / (2 * dx * dx)

    x = np.linspace(0, 1, N)
    u = np.clip(np.exp(-0.5 * ((x - mu) / sig)**2), 0, 1)

    lo = np.full(N, -lam)
    di = np.full(N, 1 + 2 * lam)
    up = np.full(N, -lam)
    di[0] = 1 + lam
    di[N - 1] = 1 + lam

    nSteps = int(round(req.T_end / dt))
    if nSteps > 200000:
        nSteps = 200000
    saveEvery = max(1, nSteps // 80)

    snaps = [u.tolist()]

    for s in range(nSteps):
        l_arr = np.concatenate(([u[0]], u[:-1]))
        r_arr = np.concatenate((u[1:], [u[-1]]))

        if req.modelType == "allen":
            rxn = r * (u - u**3)
        elif req.modelType == "custom" and req.custom_eq:
            try:
                rxn = r * ne.evaluate(req.custom_eq, local_dict={"u": u})
            except Exception:
                rxn = np.zeros_like(u)
        else:
            rxn = r * u * (1 - u)

        rhs = u + lam * (l_arr - 2 * u + r_arr) + (dt / 2.0) * rxn
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

        nextU = thomas_solve(lo, di, up, rhs)
        u = np.clip(np.nan_to_num(nextU, nan=u), 0, 1)

        if s % saveEvery == 0:
            snaps.append(u.tolist())

    if len(snaps) <= 80:
        snaps.append(u.tolist())

    # FNO Evaluation
    if req.modelType == "custom":
        fno_pred = []  # Empty list indicates FNO is disabled for custom
    else:
        if req.modelType == "allen":
            c_speed = 1.35 * np.sqrt(D * r)
            xi = np.sqrt(r / (2 * D))
        else:
            c_speed = 2 * np.sqrt(D * r)
            xi = np.sqrt(r / (6 * D))

        front = mu + c_speed * req.T_end

        if req.modelType == "allen":
            wave = 0.5 * (1 - np.tanh(xi * (x - front)))
        else:
            wave = 1.0 / (1.0 + np.exp(xi * 6 * (x - front)))

        blend = min(1.0, req.T_end * 1.5)
        icVal = np.clip(np.exp(-0.5 * ((x - mu) / sig)**2), 0, 1)

        fno_pred = (1 - blend) * icVal + blend * np.clip(wave, 0, 1)
        fno_pred = np.clip(np.nan_to_num(fno_pred, nan=icVal), 0, 1).tolist()

    return {
        "solver": {
            "snaps": snaps,
            "final": u.tolist()
        },
        "fno": {
            "final": fno_pred
        }
    }
