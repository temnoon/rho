# Rho Small‑Embedding Demo — Humanizer Node

A transparent, working demonstration of a 64‑dimensional “small embedding” density matrix (ρ) as a model of **subjective personification** in a lexical field. This document includes philosophy, math, architecture, APIs, UI, code, and deployment.

---

## 0) What we’re building (in one glance)

- **Goal:** A public node on your domain (e.g., `rho.humanizer.com`) where visitors can *see and manipulate* an AI Personified Reader’s state ρ as it reads narratives.
- **Core:** 64‑D local meaning space; ρ is a density matrix updated by reading operations; POVM packs probe interpretable attributes (narrator distance, reliability, affect, arousal, myth/realism, agency, etc.).
- **Transparency:** Every operation shows: inputs → math → outputs (ρ heatmap, eigenvalues, POVM probabilities, change deltas, explanations).
- **Teaching mode:** Built‑in guide and an **Agent‑Helper** that recommends which texts/operations best demonstrate particular properties.
- **Stack:** FastAPI (Python, NumPy) for math + embedding bridge; React UI (Tailwind + shadcn/ui + Recharts) for visualization; Docker for deployment; optional Discourse embedding.

---

## 1) Philosophical framing: Personification in the Lexical Field

- **Personification (site vocabulary):** the modeled *person* as an **agent role** moving through a lexical field (the shared space of text/media/discourse). The biological human *hosts* the role, but in the field, the **person** is the relevant agent.
- **Lexical field:** exists only in participation; meanings are enacted and negotiated. (Alan Watts’ “person” as *mask/role*—Greek theatre.)
- **ρ (rho):** the **momentary subjective stance** of the Personified Reader. Mathematically a density matrix on a local 64‑D meaning space.
- **POVMs:** measurement operators that produce *readable* attribute probabilities (e.g., narrator distance ±). These are the dials we show to visitors.
- **Husserlian nod:** treat ρ as Noema (structured sense), with reading/writing as Noesis (acts). Reading updates ρ; measuring probes it; style/namespace/persona act as unitary‑like transforms.

**Copy for the landing panel (concise):**

> “Here you can watch an AI Personified Reader encounter text. Its internal stance is a 64‑D state ρ. As it reads, ρ shifts. Probes (POVMs) reveal interpretable attributes like tone, narrator distance, and reliability. You can load sample texts, apply operations, and see the math that links words to meaning.”

---

## 2) Math model (minimal, inspectable)

### 2.1 Local meaning space

- Let the *global* embedding of a passage be \(x \in \mathbb{R}^m\) (e.g., 384–1024). We keep a **learned projection** \(W \in \mathbb{R}^{64\times m}\).
- Local vector \(v = \mathrm{norm}(Wx) \in \mathbb{R}^{64}\).

### 2.2 Density matrix construction

- Treat a fresh narrative as a **pure state** \(|v\rangle\langle v|\).
- Update rule (exponential moving blend):
  $$
  \rho_\text{new} \;=\; (1-\alpha)\,\rho_\text{old} + \alpha\, |v\rangle\langle v| \qquad (\alpha\in[0,1])
  $$
- Renormalize to \(\mathrm{Tr}(\rho)=1\) and re‑PSD via symmetric projection if needed.

### 2.3 Measurements (POVMs)

- A **POVM pack** is \(\{E_i\}_{i=1..k}\) with \(E_i\succeq 0\) and \(\sum_i E_i=I\).
- Probabilities: \(p_i = \operatorname{Tr}(E_i\,\rho)\).
- For 2‑outcome axes (e.g., *narrator distance*: near vs far) we use rank‑1 projectors plus complement: \(E_{+}=|u\rangle\langle u|,\; E_{-}=I-E_{+}\).
- Packs can be hand‑crafted (transparent) or learned (from labeled corpora). Start with **transparent** fixed bases for demo.

### 2.4 Diagnostics

- **Eigen‑spectrum:** \(\rho = Q\Lambda Q^\top\). Show \(\lambda\) bar chart, purity \(\operatorname{Tr}(\rho^2)\), entropy \( -\operatorname{Tr}(\rho\log \rho)\).
- **Deltas:** \(\Delta\rho = \rho_\text{new}-\rho_\text{old}\), \(\Delta p\) for each POVM.

---

## 3) System architecture

```
[React UI (Next.js/Vite)]
  ├─ Panels: Text Loader | Read (α) | Measure | Visualize | Explain | Agent Helper
  ├─ Visuals: ρ heatmap, eigenvalues, POVM bars, diff maps
  └─ Calls REST →
[FastAPI backend]
  ├─ /embed  : global embedding (pluggable)
  ├─ /project: W·x → v (64‑D)
  ├─ /rho     : create/get/update; read(α); reset
  ├─ /measure : apply POVM pack(s)
  ├─ /math    : eig, purity, entropy, diffs
  ├─ /packs   : list/load custom POVM packs
  └─ /explain : generate stepwise textual explanations
[Storage]
  ├─ SQLite or Postgres (ρ snapshots, logs, narratives, packs)
  └─ Files: W matrix, pack JSON, demo texts
[Edge]
  └─ Caddy/Nginx reverse proxy → TLS → /api and /app
```

---

## 4) Data schemas (JSON)

### 4.1 Rho state

```json
{
  "rho_id": "uuid",
  "dim": 64,
  "rho": [[...64 floats]...64 rows],
  "trace": 1.0,
  "purity": 0.0,
  "entropy": 0.0,
  "eigs": {"vals": [...], "vecs_ref": "file://...optional"},
  "updated_at": "2025-08-10T00:00:00Z",
  "metadata": {"label": "Visitor A", "notes": "demo"}
}
```

### 4.2 POVM pack

```json
{
  "pack_id": "axes_12x2_demo",
  "axes": [
    {"id": "narrator_distance", "u": [64 floats], "labels": ["near","far"]},
    {"id": "reliability",       "u": [64 floats], "labels": ["plus","minus"]},
    {"id": "affect_valence",     "u": [64 floats], "labels": ["positive","negative"]}
    // ...
  ]
}
```

### 4.3 Narrative record

```json
{
  "text_id": "uuid",
  "title": "Tortoise and Hare (Paraphrase)",
  "namespace": "Aesop",
  "persona": "omniscient fable teller",
  "style": "plain",
  "text": "Once a boastful hare...",
  "embedding_ref": "computed on demand",
  "tags": ["demo","fable"]
}
```

---

## 5) REST API (FastAPI)

### 5.1 Endpoints

- `POST /rho/init` → create new ρ (identity/thermal or from seed text)
- `GET /rho/{rho_id}` → fetch current ρ + diagnostics
- `POST /rho/{rho_id}/read` → body: `{ text_id | raw_text, alpha }`
- `POST /rho/{rho_id}/measure` → body: `{ pack_id }` → returns axis probs
- `POST /rho/{rho_id}/reset` → identity/seed
- `GET /packs` / `POST /packs` → list/add custom packs
- `POST /math/eig` → eigendecomposition of current ρ
- `POST /explain` → structured explanation of last operation

### 5.2 Python (FastAPI) — single‑file reference implementation

```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json, time, uuid

app = FastAPI(title="Humanizer Rho Demo")
DIM = 64

# --- storage (in‑memory for demo) ---
STATE = {}
PACKS = {}

# --- models ---
class ReadReq(BaseModel):
    raw_text: str | None = None
    text_id: str | None = None
    alpha: float = 0.2

class MeasureReq(BaseModel):
    pack_id: str

# --- helpers ---

def psd_project(A):
    # symmetrize, clip negative eigs
    A = 0.5*(A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0, None)
    A = (V * w) @ V.T
    tr = np.trace(A)
    if tr <= 0: # fallback
        A = np.eye(DIM)/DIM
    else:
        A /= tr
    return A

# placeholder projection W: orthonormal 64x64 for demo
np.random.seed(7)
Q, _ = np.linalg.qr(np.random.randn(DIM, DIM))
W = Q

# simple deterministic embedding stub (replace with real model)

def embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    x = rng.standard_normal(DIM)
    x /= np.linalg.norm(x) + 1e-9
    return x

# --- routes ---
@app.post("/rho/init")
def rho_init():
    rho_id = str(uuid.uuid4())
    rho = np.eye(DIM)/DIM
    STATE[rho_id] = {"rho": rho, "log": []}
    return {"rho_id": rho_id, **diagnostics(rho)}

@app.get("/rho/{rho_id}")
def rho_get(rho_id: str):
    rho = STATE[rho_id]["rho"]
    return {"rho_id": rho_id, **diagnostics(rho)}

@app.post("/rho/{rho_id}/read")
def rho_read(rho_id: str, req: ReadReq):
    assert req.raw_text or req.text_id
    text = req.raw_text or f"[text_id:{req.text_id}]"
    v = embed(text)  # already 64‑D unit
    rho = STATE[rho_id]["rho"]
    pure = np.outer(v, v)
    rho_new = (1-req.alpha)*rho + req.alpha*pure
    rho_new = psd_project(rho_new)
    STATE[rho_id]["rho"] = rho_new
    log = STATE[rho_id]["log"]
    log.append({"ts": time.time(), "op": "read", "alpha": req.alpha, "text": text})
    return {"rho_id": rho_id, **diagnostics(rho_new)}

@app.post("/rho/{rho_id}/measure")
def rho_measure(rho_id: str, req: MeasureReq):
    rho = STATE[rho_id]["rho"]
    pack = PACKS[req.pack_id]
    probs = {}
    I = np.eye(DIM)
    for axis in pack["axes"]:
        u = np.array(axis["u"], dtype=float)
        u /= np.linalg.norm(u) + 1e-9
        E_plus = np.outer(u, u)
        E_minus = I - E_plus
        p_plus = float(np.trace(E_plus @ rho))
        p_minus = float(np.trace(E_minus @ rho))
        probs[axis["id"]] = {
            axis["labels"][0]: p_plus,
            axis["labels"][1]: p_minus
        }
    return {"rho_id": rho_id, "pack_id": req.pack_id, "probs": probs}

@app.get("/packs")
def packs_list():
    return {"packs": list(PACKS.keys())}

@app.post("/packs")
def packs_add(body: dict):
    PACKS[body["pack_id"]] = body
    return {"ok": True, "pack_id": body["pack_id"]}

# --- diagnostics ---

def diagnostics(rho: np.ndarray):
    w = np.linalg.eigvalsh(0.5*(rho+rho.T))
    w = np.clip(w, 1e-12, None)
    purity = float(np.sum(w*w))
    entropy = float(-np.sum(w*np.log(w)))
    return {
        "dim": DIM,
        "trace": float(np.trace(rho)),
        "purity": purity,
        "entropy": entropy,
        "eigs": sorted([float(x) for x in w], reverse=True)
    }
```

> Swap `embed()` with your local embedding bridge (e.g., bge‑small, MiniLM, E5). If the global embedding has m≠64, multiply by a learned `W` (persisted to disk) before normalization.

---

## 6) Frontend (React) — minimal but polished

- **Libraries:** Tailwind CSS, `shadcn/ui`, `lucide-react`, `recharts`.
- **Panels:**
  1. **Narrative Loader** (demo texts + paste box)
  2. **Read** (α slider) → updates ρ
  3. **Measure** (select pack) → bar charts
  4. **Visualize ρ** (heatmap + eigenvalues)
  5. **Explain** (stepwise explanation log)
  6. **Agent Helper** (suggests texts/ops for targeted effects)

### 6.1 React component (single file demo)

```tsx
// app/RhoDemo.tsx
import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const API = "/api"; // proxy to FastAPI

export default function RhoDemo() {
  const [rhoId, setRhoId] = useState<string | null>(null);
  const [a
```
