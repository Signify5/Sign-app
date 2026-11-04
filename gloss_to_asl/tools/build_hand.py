import gzip
import json
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
MH = ASSETS / "makehuman"
OUT = ASSETS / "hand" / "makehuman_hand.json"
SHAPE = MH / "mpfb" / "targets" / "caucasian-male-young.target.gz"

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
    return np.stack([x, y, z])


def top4(W):
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w /= w.sum(1, keepdims=True)
    return idx, w


def mesh_tips(V, W, joints):
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


def read_obj(obj):
    V, groups, group = [], {}, None
    for line in open(obj, encoding="utf-8"):
        if line.startswith("v "):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            groups.setdefault(group, []).append(idx)
    V = np.array(V)
    if Path(obj).name == "base.obj" and SHAPE.exists():
        V = V + torso_shape(V)
    return V, groups


def _ramp(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def torso_shape(V):
    d = np.zeros_like(V)
    with gzip.open(SHAPE, "rt") as f:
        for line in f:
            t = line.split()
            if len(t) == 4:
                d[int(t[0])] = [float(x) for x in t[1:]]
    d[:, 1] = 0
    w = _ramp(V[:, 1], 1.5, 3.0) * (1 - _ramp(V[:, 1], 5.4, 5.9)) * (1 - _ramp(np.abs(V[:, 0]), 1.6, 2.4))
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    waist = _ramp(y, 1.3, 2.2) * (1 - _ramp(y, 3.0, 3.8)) * (1 - _ramp(np.abs(x), 1.8, 2.6))
    chest = _ramp(y, 3.4, 4.2) * (1 - _ramp(y, 5.3, 5.8)) * (1 - _ramp(np.abs(x), 2.4, 3.4))
    pec = _ramp(y, 3.8, 4.3) * (1 - _ramp(y, 5.0, 5.4)) * _ramp(np.abs(x), 0.1, 0.5) * (1 - _ramp(np.abs(x), 1.3, 1.9)) * (z > 0)
    dx = x * (0.07 * chest - 0.02 * waist)
    dz = 0.1 * pec
    return d * w[:, None] + np.stack([dx, 0 * x, dz], 1) + arm_bulk(V)


ARM_PTS = np.array([[2.09, 5.25, 0.22], [3.13, 3.49, 0.13], [4.31, 2.45, 1.76]])


def arm_bulk(V, grow=0.22):
    P = V.copy()
    P[:, 0] = np.abs(V[:, 0])
    best_r = np.full(len(V), 9.0)
    best_t = np.zeros(len(V))
    best_c = np.zeros_like(V)
    total = 0.0
    for a, b in zip(ARM_PTS[:-1], ARM_PTS[1:]):
        ab = b - a
        L = np.linalg.norm(ab)
        u = np.clip(((P - a) @ ab) / (L * L), 0, 1)
        c = a + u[:, None] * ab
        r = np.linalg.norm(P - c, axis=1)
        m = r < best_r
        best_r[m] = r[m]
        best_t[m] = (total + u[m] * L) / 4.31
        best_c[m] = c[m]
        total += L
    along = _ramp(best_t, 0.12, 0.3) * (1 - _ramp(best_t, 0.78, 0.95))
    near = 1 - _ramp(best_r, 0.7, 1.1)
    out = (P - best_c) * (grow * along * near)[:, None]
    out[:, 0] *= np.sign(V[:, 0] + 1e-9)
    return out


class MakeHumanHand:
    """The MakeHuman right hand: wrist frame, scale and the part of the body mesh that belongs to the hand."""

    def __init__(self, Vall, F, skel, weights):
        self.V, self.F = Vall, F
        s = self.skel = json.load(open(skel))
        wj = self.wj = json.load(open(weights))["weights"]
        self.jpos = lambda name: Vall[s["joints"][name]].mean(0)
        self.head = lambda b: self.jpos(s["bones"][b]["head"])
        self.tail = lambda b: self.jpos(s["bones"][b]["tail"])
        head, tail = self.head, self.tail

        mh = {}
        for i, f in enumerate(FINGERS):
            mh[f"{f}-mc"] = f"metacarpal{i+1}.R"
            for k, part in enumerate(("mcp", "pip", "dip")):
                mh[f"{f}-{part}"] = f"finger{i+2}-{k+1}.R"
        for k, part in enumerate(("cmc", "mcp", "ip")):
            mh[f"thumb-{part}"] = f"finger1-{k+1}.R"
        self.mh = mh
        self.joints_old = {"wrist": head("wrist.R")}
        for n, b in mh.items():
            self.joints_old[n] = head(b)
        self.tips_old = {f: tail(f"finger{i+2}-3.R") for i, f in enumerate(FINGERS)}
        self.tips_old["thumb"] = tail("finger1-3.R")

        n = len(Vall)
        W = np.zeros((n, len(NAMES)))
        for n_, b in mh.items():
            for vi, w in wj.get(b, []):
                W[vi, JI[n_]] += w
        for b in ("wrist.R", "lowerarm02.R", "lowerarm01.R"):
            for vi, w in wj.get(b, []):
                W[vi, 0] += w
        self.W = W

        jo = self.joints_old
        self.R0 = frame(jo["wrist"], jo["middle-mcp"], jo["thumb-mcp"])
        self.wr = jo["wrist"]
        hand_len = np.linalg.norm((self.tips_old["middle"] - self.wr) @ self.R0.T)
        self.scale = 0.19 / hand_len
        self.Vn = (Vall - self.wr) @ self.R0.T * self.scale
        arm_dir = -(self.wr - head("lowerarm02.R")) @ self.R0.T
        self.arm_dir = arm_dir / np.linalg.norm(arm_dir)
        self.keep_v = (W.sum(1) > 0.5) & ((self.Vn @ self.arm_dir) < 0.09)


def build_makehuman(obj, skel, weights):
    Vall, groups = read_obj(obj)
    F = groups["body"]
    h = MakeHumanHand(Vall, F, skel, weights)
    W, Vn, scale = h.W.copy(), h.Vn, h.scale
    n = len(Vall)
    faces = []
    for f in F:
        if all(h.keep_v[i] for i in f):
            faces.append(f[:3]); faces.append([f[0], f[2], f[3]]) if len(f) == 4 else None
    faces = np.array(faces)
    used = np.unique(faces)
    remap = -np.ones(n, int); remap[used] = np.arange(len(used))
    V = Vn[used]; Wk = W[used]; Fk = remap[faces]
    Wk /= Wk.sum(1, keepdims=True)
    to = lambda p: (p - h.wr) @ h.R0.T * scale
    joints = np.array([to(h.joints_old[nm]) for nm in NAMES])
    tips = {k: to(v) for k, v in h.tips_old.items()}
    return pack("MakeHuman", V, Fk, Wk, joints, tips, 0.19)


if __name__ == "__main__":
    hand = build_makehuman(MH / "base.obj", MH / "default.mhskel", MH / "default_weights.mhw")
    OUT.write_text(json.dumps(hand, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KB")
