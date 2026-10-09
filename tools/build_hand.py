"""MakeHuman base mesh (CC0) right hand -> assets/hand/makehuman_hand.json, the rig the 3D player skins.

Run from the repo root: python tools/build_hand.py

Output frame: metres, wrist at origin, fingers along +y, palm facing +z, thumb toward +x (right hand).
Unified joints (20): 0 wrist; index/middle/ring/pinky each [mc, mcp, pip, dip]; thumb [cmc, mcp, ip].
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MH = ROOT / "assets" / "makehuman"
OUT = ROOT / "assets" / "hand" / "makehuman_hand.json"

FINGERS = ["index", "middle", "ring", "pinky"]
NAMES = ["wrist"] + [f"{f}-{p}" for f in FINGERS for p in ("mc", "mcp", "pip", "dip")] + ["thumb-cmc", "thumb-mcp", "thumb-ip"]
PARENT = {"wrist": None, "thumb-cmc": "wrist", "thumb-mcp": "thumb-cmc", "thumb-ip": "thumb-mcp"}
for f in FINGERS:
    PARENT.update({f"{f}-mc": "wrist", f"{f}-mcp": f"{f}-mc", f"{f}-pip": f"{f}-mcp", f"{f}-dip": f"{f}-pip"})
JI = {n: i for i, n in enumerate(NAMES)}


def frame(wrist, mid_mcp, thumb_ref):
    y = mid_mcp - wrist; y /= np.linalg.norm(y)
    x = thumb_ref - wrist; x -= y * x.dot(y); x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z])  # rows: new axes in old coords


def top4(W):
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w /= w.sum(1, keepdims=True)
    return idx, w


def mesh_tips(V, W, joints):
    """Fingertip = furthest point along the distal segment's principal axis (vertices owned by the last joint)."""
    tips = {}
    for f, last in [("thumb", "thumb-ip"), ("index", "index-dip"), ("middle", "middle-dip"),
                    ("ring", "ring-dip"), ("pinky", "pinky-dip")]:
        j = JI[last]
        P = V[W.argmax(1) == j]
        D = joints[j]
        ref = D - joints[JI[PARENT[last]]]
        ax = np.linalg.svd(P - P.mean(0))[2][0]
        ax *= np.sign(ax @ ref)
        tips[f] = D + ax * ((P - D) @ ax).max()
    return tips


def pack(name, V, F, W, joints, tips, scale, extra=None):
    tips = mesh_tips(V, W, joints)
    idx, w = top4(W)
    out = {
        "name": name,
        "verts": np.round(V.ravel(), 5).tolist(),
        "faces": F.astype(int).ravel().tolist(),
        "skinIdx": idx.astype(int).ravel().tolist(),
        "skinW": np.round(w.ravel(), 4).tolist(),
        "joints": np.round(joints, 5).tolist(),
        "parents": [JI[PARENT[n]] if PARENT[n] else -1 for n in NAMES],
        "names": NAMES,
        "tips": {k: np.round(v, 5).tolist() for k, v in tips.items()},
        "hasMc": extra.pop("hasMc", True) if extra else True,
    }
    if extra:
        out.update(extra)
    print(f"{name}: {len(V)} verts, {len(F)} faces, hand length {scale*1000:.0f} mm")
    return out


def build_makehuman(obj, skel, weights):
    Vall, F = [], []
    group = None
    for line in open(obj, encoding="utf-8"):
        if line.startswith("v "):
            Vall.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
        elif line.startswith("f ") and group == "body":
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    Vall = np.array(Vall)
    s = json.load(open(skel)); wj = json.load(open(weights))["weights"]
    jpos = lambda name: Vall[s["joints"][name]].mean(0)
    head = lambda b: jpos(s["bones"][b]["head"])
    tail = lambda b: jpos(s["bones"][b]["tail"])

    # MakeHuman: finger1 = thumb ... finger5 = pinky; metacarpal1..4 = index..pinky.
    mh = {}
    for i, f in enumerate(FINGERS):
        mh[f"{f}-mc"] = f"metacarpal{i+1}.R"
        for k, part in enumerate(("mcp", "pip", "dip")):
            mh[f"{f}-{part}"] = f"finger{i+2}-{k+1}.R"
    for k, part in enumerate(("cmc", "mcp", "ip")):
        mh[f"thumb-{part}"] = f"finger1-{k+1}.R"
    joints_old = {"wrist": head("wrist.R")}
    for n, b in mh.items():
        joints_old[n] = head(b)
    tips_old = {f: tail(f"finger{i+2}-3.R") for i, f in enumerate(FINGERS)}
    tips_old["thumb"] = tail("finger1-3.R")

    # Vertex weights for hand bones; forearm bones fold into the wrist joint.
    n = len(Vall)
    W = np.zeros((n, len(NAMES)))
    for n_, b in mh.items():
        for vi, w in wj.get(b, []):
            W[vi, JI[n_]] += w
    hand_total = W.sum(1).copy()
    for b in ("wrist.R", "lowerarm02.R", "lowerarm01.R"):
        for vi, w in wj.get(b, []):
            W[vi, 0] += w

    R0 = frame(joints_old["wrist"], joints_old["middle-mcp"], joints_old["thumb-mcp"])
    wr = joints_old["wrist"]
    # Keep body faces whose vertices are hand/wrist-weighted, cut ~9 cm up the forearm.
    Vn = (Vall - wr) @ R0.T
    hand_len = np.linalg.norm((tips_old["middle"] - wr) @ R0.T)
    scale = 0.19 / hand_len
    Vn *= scale
    arm_dir = -(joints_old["wrist"] - head("lowerarm02.R")) @ R0.T
    arm_dir /= np.linalg.norm(arm_dir)
    keep_v = (W.sum(1) > 0.5) & ((Vn @ arm_dir) < 0.09)
    faces = []
    for f in F:
        if all(keep_v[i] for i in f):
            faces.append(f[:3]); faces.append([f[0], f[2], f[3]]) if len(f) == 4 else None
    faces = np.array(faces)
    used = np.unique(faces)
    remap = -np.ones(n, int); remap[used] = np.arange(len(used))
    V = Vn[used]; Wk = W[used]; Fk = remap[faces]
    Wk /= Wk.sum(1, keepdims=True)
    to = lambda p: (p - wr) @ R0.T * scale
    joints = np.array([to(joints_old[nm]) for nm in NAMES])
    tips = {k: to(v) for k, v in tips_old.items()}
    return pack("MakeHuman", V, Fk, Wk, joints, tips, 0.19)


if __name__ == "__main__":
    hand = build_makehuman(MH / "base.obj", MH / "default.mhskel", MH / "default_weights.mhw")
    OUT.write_text(json.dumps(hand, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KB")
