"""Builds the upper-body avatar (body, shirt, hair, brows, eyes) around the MakeHuman hand.

Inputs (CC0, from https://github.com/makehumancommunity/makehuman, makehuman/data) go in assets/makehuman/:
  base.obj, default.mhskel, default_weights.mhw, eyes/high-poly/high-poly.{obj,mhclo}, eyes/materials/brown_eye.png
Output: assets/hand/makehuman_body.json
Everything except the base mesh and the eye model (shirt, hair, brows, skin and eye shading) is generated here.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from gloss_to_asl.hand3d import _blob
from gloss_to_asl.tools.build_hand import MakeHumanHand, read_obj

ASSETS = Path(__file__).resolve().parent.parent / "assets"
MH = ASSETS / "makehuman"
OUT = ASSETS / "hand" / "makehuman_body.json"

BONES = ["torso", "neck", "head"] + [f"{p}{s}" for s in "RL" for p in ("up", "f0", "f1", "f2", "wr")] + ["jaw", "clR", "clL"]
BI = {n: i for i, n in enumerate(BONES)}
HAND_CUT = -0.026
CUFF = 0.03
HIP_CUT_DM = 0.8
HEM_DM = 0.9
POS_Q = 2e4
HEAD_WORDS = ("head", "jaw", "eye", "levator", "oculi", "orbicularis", "oris", "risorius", "special", "tongue", "temporalis")


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def vertex_normals(V, T):
    n = np.zeros_like(V)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3):
        np.add.at(n, T[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def compact(V, T, *attrs):
    used = np.unique(T)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return (V[used], remap[T]) + tuple(a[used] for a in attrs)


def top4(W):
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    return idx, w


def loop_subdivide(V, T, levels):
    """Loop subdivision with the usual boundary rules (the hair cap is an open patch)."""
    for _ in range(levels):
        n = len(V)
        edge_id, opp, ends = {}, [], []
        for a, b, c in T:
            for p, q, o in ((a, b, c), (b, c, a), (c, a, b)):
                k = (min(p, q), max(p, q))
                if k not in edge_id:
                    edge_id[k] = len(ends)
                    ends.append(k)
                    opp.append([])
                opp[edge_id[k]].append(o)
        nbr = [set() for _ in range(n)]
        bnd = [[] for _ in range(n)]
        for (p, q), o in zip(ends, opp):
            nbr[p].add(q)
            nbr[q].add(p)
            if len(o) == 1:
                bnd[p].append(q)
                bnd[q].append(p)
        newV = np.zeros((n + len(ends), V.shape[1]))
        for i in range(n):
            if len(bnd[i]) >= 2:
                newV[i] = 0.75 * V[i] + 0.125 * (V[bnd[i][0]] + V[bnd[i][1]])
            else:
                k = len(nbr[i])
                beta = 3 / 16 if k == 3 else 3 / (8 * k)
                newV[i] = (1 - k * beta) * V[i] + beta * V[list(nbr[i])].sum(0)
        for e, ((p, q), o) in enumerate(zip(ends, opp)):
            newV[n + e] = (0.375 * (V[p] + V[q]) + 0.125 * (V[o[0]] + V[o[1]])) if len(o) == 2 else 0.5 * (V[p] + V[q])
        nt = []
        for a, b, c in T:
            ab, bc, ca = (n + edge_id[(min(x, y), max(x, y))] for x, y in ((a, b), (b, c), (c, a)))
            nt += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V, T = newV, np.array(nt)
    return V, T


def cavity(P, T):
    """0..1 per vertex: how much the surface sits in a pit at three scales (eye sockets, nostrils, folds)."""
    import scipy.sparse as sp
    n = len(P)
    r = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    c = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1
    A = sp.diags(1 / np.maximum(A.sum(1).A1, 1)) @ A
    N = vertex_normals(P, T)
    out = np.zeros(n)
    X = P.copy()
    done = 0
    for k, scale, w in ((2, 0.0012, 0.5), (6, 0.0030, 0.35), (16, 0.0060, 0.25)):
        for _ in range(k - done):
            X = A @ X
        done = k
        out += w * np.clip(((X - P) * N).sum(1) / scale, 0, 1)
    return np.clip(out, 0, 1)


def drape(S, P, N, off, T, Vdm, iters=14):
    """Smooth the shirt over the belly and chest, but never let it sink into the skin: it fills pits like the navel."""
    n = len(S)
    nbr = [set() for _ in range(n)]
    for a, b, c in T:
        nbr[a] |= {b, c}
        nbr[b] |= {a, c}
        nbr[c] |= {a, b}
    w = smoothstep(1.6, 1.1, np.abs(Vdm[:, 0])) * smoothstep(5.4, 4.6, Vdm[:, 1])
    idx = [i for i in np.flatnonzero(w > 0) if nbr[i]]
    S = S.copy()
    for _ in range(iters):
        avg = S.copy()
        for i in idx:
            avg[i] = S[list(nbr[i])].mean(0)
        S = S + (avg - S) * (0.6 * w)[:, None]
        d = ((S - P) * N).sum(1)
        S = S + N * np.maximum(off - d, 0)[:, None]
    return S


def round_shoulders(P, T, centres, radius=0.08, iters=10):
    """Soften the squared-off shoulder caps of the base mesh with a local Laplacian smooth."""
    n = len(P)
    nbr = [set() for _ in range(n)]
    for a, b, c in T:
        nbr[a] |= {b, c}
        nbr[b] |= {a, c}
        nbr[c] |= {a, b}
    d = np.min([np.linalg.norm(P - c, axis=1) for c in centres], axis=0)
    w = smoothstep(radius, radius * 0.45, d)
    w *= smoothstep(centres[0][1] - 0.04, centres[0][1] + 0.01, P[:, 1])
    idx = [i for i in np.flatnonzero(w > 0) if nbr[i]]
    P = P.copy()
    for _ in range(iters):
        avg = P.copy()
        for i in idx:
            avg[i] = P[list(nbr[i])].mean(0)
        P = P + (avg - P) * (0.55 * w)[:, None]
    return P


def boundary_edges(T):
    seen = {}
    for a, b, c in T:
        for p, q in ((a, b), (b, c), (c, a)):
            seen.setdefault((min(p, q), max(p, q)), []).append((p, q))
    return [e[0] for e in seen.values() if len(e) == 1]


class Body:
    def __init__(self):
        V, groups = read_obj(MH / "base.obj")
        self.Vdm = V
        self.hand = MakeHumanHand(V, groups["body"], MH / "default.mhskel", MH / "default_weights.mhw")
        h = self.hand
        self.s = h.scale
        self.ms = (h.head("upperarm01.R") + h.head("upperarm01.L")) / 2
        self.scene = lambda P: (np.asarray(P) - self.ms) * self.s
        quads = np.array(groups["body"])
        self.tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        mirror = cKDTree(V).query(V * [-1, 1, 1])[1]
        self.mirror = mirror
        left_side = V[:, 0] > 1e-4
        self.side = np.where(left_side, 1, 0)
        src = np.where(left_side, mirror, np.arange(len(V)))
        self.ylocal = h.Vn[src, 1]
        self.inhand = h.keep_v[src]
        self.groups = groups

    def weights(self):
        """Dense (n, 13) skin weights on the simplified bones."""
        h, V = self.hand, self.Vdm
        n = len(V)
        W = np.zeros((n, len(BONES)))
        forearm = np.zeros(n)
        self.breast = np.zeros(n)
        raw = json.load(open(MH / "default_weights.mhw"))["weights"]
        for name, lst in raw.items():
            side = name[-1] if name[-2:] in (".R", ".L") else None
            base = name[:-2] if side else name
            idx = np.array([i for i, _ in lst], int)
            w = np.array([x for _, x in lst], float)
            arm = lambda p: BI[f"{p}{side}"] if side else None
            if base == "breast":
                np.add.at(self.breast, idx, w)
            if base.startswith(("lowerarm", "wrist", "finger", "metacarpal")):
                np.add.at(forearm, idx, w)
            elif base.startswith("upperarm"):
                np.add.at(W[:, arm("up")], idx, w)
            elif base == "clavicle":
                np.add.at(W[:, arm("cl")], idx, w)
            elif base == "shoulder01":
                np.add.at(W[:, arm("up")], idx, w * 0.5)
                np.add.at(W[:, arm("cl")], idx, w * 0.5)
            elif base == "neck01":
                np.add.at(W[:, 0], idx, w * 0.5)
                np.add.at(W[:, 1], idx, w * 0.5)
            elif base == "neck02":
                np.add.at(W[:, 1], idx, w)
            elif base == "neck03":
                np.add.at(W[:, 1], idx, w * 0.5)
                np.add.at(W[:, 2], idx, w * 0.5)
            elif base in ("jaw", "oris01"):
                np.add.at(W[:, BI["jaw"]], idx, w)
            elif base.startswith(("oris03", "risorius")):
                np.add.at(W[:, BI["jaw"]], idx, w * 0.5)
                np.add.at(W[:, 2], idx, w * 0.5)
            elif base.startswith(HEAD_WORDS):
                np.add.at(W[:, 2], idx, w)
            else:
                np.add.at(W[:, 0], idx, w)
        E, Wr = h.head("lowerarm01.R"), h.head("wrist.R")
        axis = Wr - E
        L = np.linalg.norm(axis)
        axis /= L
        P = V.copy()
        P[self.side == 1, 0] *= -1
        t = (P - E) @ axis / L
        nodes = self.nodes = [0.0, 0.3, 0.6, 1 - 0.026 / (L * self.s)]
        seg = np.clip(np.searchsorted(nodes, t, side="right") - 1, 0, 2)
        a = np.clip((t - np.take(nodes, seg)) / (np.take(nodes, seg + 1) - np.take(nodes, seg)), 0, 1)
        fw = np.zeros((n, 4))
        fw[np.arange(n), seg] += 1 - a
        fw[np.arange(n), seg + 1] += a
        for k, bone in enumerate(("f0", "f1", "f2", "wr")):
            for letter, m in (("R", self.side == 0), ("L", self.side == 1)):
                W[m, BI[f"{bone}{letter}"]] += (forearm * fw[:, k])[m]
        # forearm weights on the centre line can only come from a side bone; assign by x
        # split the lips cleanly at the mouth line so the jaw can open them
        box = (np.abs(V[:, 0]) < 0.62) & (V[:, 1] > 6.42) & (V[:, 1] < 6.78) & (V[:, 2] > 1.0)
        g = smoothstep(6.616, 6.598, V[:, 1]) * smoothstep(0.62, 0.30, np.abs(V[:, 0]))
        keep = np.maximum(W[:, BI["jaw"]], g)
        W[box] = 0
        W[box, BI["jaw"]] = keep[box]
        W[box, 2] = 1 - keep[box]
        W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
        self.t_forearm = t / 3
        self.forearm_w = forearm
        return W


def build():
    B = Body()
    h, V = B.hand, B.Vdm
    n = len(V)

    inhand_tri = B.inhand[B.tris].all(1) & (B.ylocal[B.tris].min(1) > HAND_CUT)
    above = V[B.tris][:, :, 1].mean(1) > HIP_CUT_DM
    keep = ~inhand_tri & above
    T0 = B.tris[keep]
    Wd = B.weights()
    P = B.scene(V)
    Nrm = vertex_normals(P, T0)
    P = P - Nrm * (0.022 * np.clip(B.breast, 0, 1))[:, None]
    sh_r = B.scene(h.head("upperarm01.R"))
    P = round_shoulders(P, T0, [sh_r, sh_r * [-1, 1, 1]])
    Nrm = vertex_normals(P, T0)

    # ---- body
    Pb, Tb, Wb = compact(P, T0, Wd)
    used_idx = np.unique(T0)
    print("body:", len(Pb), "verts,", len(Tb), "tris")

    # ---- shirt
    headw = Wd[:, 1] + Wd[:, 2] + Wd[:, BI["jaw"]]
    y_dm = V[:, 1]
    neckline = 5.95 - 0.15 * np.clip((V[:, 2] - 0.5) / 0.9, 0, 1)
    shirt_v = (headw < 0.5) & (y_dm > HEM_DM) & (y_dm < neckline + 0.6) & (B.ylocal < CUFF)
    shirt_t = shirt_v[T0].all(1)
    Ts = T0[shirt_t]
    off = np.where(np.abs(V[:, 0]) > 1.7, 0.0032, 0.0048)
    sh = drape(P + Nrm * off[:, None], P, Nrm, off, Ts, V)
    bnd = boundary_edges(Ts)
    strips = []
    for a, b in bnd:
        strips += [[a, b, n + a], [b, n + b, n + a]]
    Pall = np.concatenate([sh, P])
    Ts_all = np.concatenate([Ts, np.array(strips, int).reshape(-1, 3)]) if strips else Ts
    Ps_, Ts_, Ws_ = compact(Pall, Ts_all, np.concatenate([Wd, Wd]))
    from scipy.spatial import cKDTree
    # collar and cuff bands: distance to the neck / wrist boundary only (not the hem)
    bound_v = np.unique(np.array(bnd).ravel()) if bnd else np.array([], int)
    bound_v = bound_v[(V[bound_v, 1] > 2.0)]
    trim = cKDTree(Pall[bound_v]).query(Ps_)[0] if len(bound_v) else np.full(len(Ps_), 0.05)
    trim_q = np.round(np.clip(trim / 0.04, 0, 1) * 255).astype("u1")
    shirt_cav = cavity(Ps_, Ts_)
    print("shirt:", len(Ps_), "verts,", len(Ts_), "tris")

    # ---- hair: smooth the head surface first, then cut it at the hairline so the edge is not jagged
    head_t = (Wd[:, 2] > 0.5)[T0].all(1) & (V[T0][:, :, 1].min(1) > 6.3)
    Ph0, Th0 = compact(P, T0[head_t])
    Hp, Th = loop_subdivide(Ph0, Th0, 2)
    Hn = vertex_normals(Hp, Th)
    Hd = Hp / B.s + B.ms
    zc = 0.4
    u = np.abs(np.arctan2(Hd[:, 0], Hd[:, 2] - zc))
    hair_line = np.interp(u, [0, 0.35, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, np.pi], [8.0, 7.98, 7.9, 7.75, 7.58, 7.5, 7.3, 6.9, 6.7])
    dh = Hd[:, 1] - hair_line
    thick = 0.0016 + 0.0085 * smoothstep(-0.05, 0.55, dh) * (1 + 0.35 * smoothstep(7.8, 8.45, Hd[:, 1]))
    Hp = Hp + Hn * thick[:, None]
    Th = Th[(dh > -0.06)[Th].any(1)]
    Hp, Th, dh = compact(Hp, Th, dh)
    hl = np.round(np.clip(dh, -1.2, 1.2) * 100).astype("i1")
    hair_strands = build_hair_strands(Hp, Th, dh, B.s)
    print("hair:", len(Hp), "verts,", len(Th), "tris")

    # ---- eyes (fitted to the base mesh with the .mhclo anchors)
    eyes = build_eyes(B)

    # ---- joints / landmarks in scene space
    J = lambda name: B.scene(h.head(name))
    right = {"shoulder": J("upperarm01.R"), "elbow": J("lowerarm01.R"), "wrist": J("wrist.R")}
    eye_c = B.scene(np.array([-0.308, 7.284, 1.245]))
    # rest orientation of the hand in the body frame: Vn = R0 (P - wrist)
    brows = build_brows(B, T0, P, Nrm)

    X = np.concatenate([Pb, Wb, (shirt_v | (V[:, 1] < HEM_DM + 0.4))[used_idx][:, None].astype(float)], 1)
    X, Tb = loop_subdivide(X, Tb, 1)
    Pb, Wb, cover = X[:, :3], X[:, 3:3 + len(BONES)], X[:, -1]
    ao = cavity(Pb, Tb)
    print("body subdivided:", len(Pb), "verts,", len(Tb), "tris")
    arrays = [
        ("bVerts", np.round(Pb * POS_Q).astype("<i2")), ("bFaces", Tb.astype("<u2")),
    ]
    bi, bw = top4(Wb)
    arrays += [("bSkinIdx", bi.astype("u1")), ("bSkinW", np.round(bw * 255).astype("u1")),
               ("bCover", np.round(cover * 255).astype("u1")), ("bAo", np.round(ao * 255).astype("u1"))]
    si, sw = top4(Ws_)
    arrays += [("sVerts", np.round(Ps_ * POS_Q).astype("<i2")), ("sFaces", Ts_.astype("<u2")),
               ("sSkinIdx", si.astype("u1")), ("sSkinW", np.round(sw * 255).astype("u1")),
               ("sTrim", trim_q), ("sCav", np.round(shirt_cav * 255).astype("u1"))]
    arrays += [("hVerts", np.round(Hp * POS_Q).astype("<i2")), ("hFaces", Th.astype("<u2")), ("hHl", hl)]
    arrays += [("gVerts", np.round(hair_strands[0] * POS_Q).astype("<i2")), ("gUv", np.round(hair_strands[1] * 65535).astype("<u2")),
               ("gFaces", hair_strands[2].astype("<u2"))]
    for k, v in build_mouth(B).items():
        arrays.append((k, v))
    for k, v in eyes["arrays"].items():
        arrays.append((k, v))
    for k, v in brows["arrays"].items():
        arrays.append((k, v))
    layout, blob = _blob(arrays)

    out = {
        "bones": BONES, "posQ": POS_Q, "layout": layout, "blob": blob,
        "shoulder": right["shoulder"].tolist(), "elbow": right["elbow"].tolist(), "wrist": right["wrist"].tolist(),
        "R0": h.R0.tolist(), "wristHand": B.scene(h.wr).tolist(),
        "neckPivot": J("neck01").tolist(), "headPivot": J("head").tolist(),
        "eyeR": eye_c.tolist(), "eyeL": (eye_c * [-1, 1, 1]).tolist(),
        "eyeRadius": eyes["radius"], "eyePng": eyes["png"],
        "jawPivot": J("jaw").tolist(), "clavicle": J("clavicle.R").tolist(),
        "nodes": B.nodes, "mouthY": float(B.scene([0, 6.605, 0])[1]),
        "browY": brows["meta"],
    }
    return out


def parse_mhclo(path):
    lines = [l.rstrip() for l in open(path, encoding="utf-8") if not l.startswith("#")]
    scales, rows, mode = {}, [], None
    for l in lines:
        if not l.strip():
            continue
        t = l.split()
        if t[0] in ("x_scale", "y_scale", "z_scale"):
            scales[t[0][0]] = (int(t[1]), int(t[2]), float(t[3]))
        elif t[0] == "verts":
            mode = "verts"
        elif t[0] in ("delete_verts", "weights"):
            mode = None
        elif mode == "verts" and len(t) in (9, 1):
            if len(t) == 9:
                rows.append([float(x) for x in t])
    return scales, np.array(rows)


def recolor_eye(img):
    """The stock iris is red-brown; shift it to a natural dark brown and take the pink out of the whites."""
    px = np.asarray(img, float) / 255
    out = px.copy()
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    mx, mn = px[..., :3].max(-1), px[..., :3].min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    iris = smoothstep(0.30, 0.55, sat)
    gray = (r + g + b) / 3
    brown = np.stack([gray * 1.65 + 0.03, gray * 1.05, gray * 0.62], -1) * 0.85
    mixed = px[..., :3] * (1 - iris[..., None]) + brown * iris[..., None] * 1.0
    white = (1 - iris)[..., None]
    mixed = mixed * (1 - 0.0 * white)
    mixed = mixed * (1 - white * 0.6) + (mixed.mean(-1, keepdims=True) * 1.0 + 0.04) * white * 0.6
    out[..., :3] = np.clip(mixed, 0, 1)
    return Image.fromarray((out * 255).astype("uint8"), "RGBA")


def build_eyes(B):
    d = MH / "eyes" / "high-poly"
    scales, rows = parse_mhclo(d / "high-poly.mhclo")
    V = B.Vdm
    sc = {k: abs(V[a, "xyz".index(k)] - V[b, "xyz".index(k)]) / den for k, (a, b, den) in scales.items()}
    verts = (V[rows[:, 0].astype(int)] * rows[:, 3:4] + V[rows[:, 1].astype(int)] * rows[:, 4:5]
             + V[rows[:, 2].astype(int)] * rows[:, 5:6] + rows[:, 6:9] * [sc["x"], sc["y"], sc["z"]])
    F, uvs = [], []
    for l in open(d / "high-poly.obj", encoding="utf-8"):
        if l.startswith("vt "):
            uvs.append([float(x) for x in l.split()[1:3]])
        elif l.startswith("f "):
            F.append([[int(x) - 1 for x in t.split("/")[:2]] for t in l.split()[1:]])
    uvs = np.array(uvs)
    # one vertex per (position, uv) pair so the texture coordinates survive
    key, vpos, vuv, tris = {}, [], [], []
    for f in F:
        ids = []
        for vi, ti in f:
            k = (vi, ti)
            if k not in key:
                key[k] = len(vpos)
                vpos.append(verts[vi])
                vuv.append(uvs[ti])
            ids.append(key[k])
        tris += [ids[:3]] + ([[ids[0], ids[2], ids[3]]] if len(ids) == 4 else [])
    vpos = B.scene(np.array(vpos))
    vuv = np.array(vuv)
    tris = np.array(tris)
    right = vpos[:, 0] < 0
    centre = vpos[right].mean(0)
    img = recolor_eye(Image.open(MH / "eyes" / "materials" / "brown_eye.png").convert("RGBA").resize((256, 256), Image.LANCZOS))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    # the right eye mesh is stored once; the left eye mirrors it, so keep only the right one
    keep_v = right
    remap = -np.ones(len(vpos), int)
    remap[keep_v] = np.arange(keep_v.sum())
    tk = keep_v[tris].all(1)
    radius = float(np.linalg.norm(vpos[right] - centre, axis=1).max())
    print("eye: centre", np.round(centre, 4), "radius", round(radius, 4), keep_v.sum(), "verts")
    return {
        "arrays": {"eVerts": np.round((vpos[keep_v] - centre) * POS_Q).astype("<i2"),
                   "eUv": np.round(vuv[keep_v] * 65535).astype("<u2"), "eFaces": remap[tris[tk]].astype("<u2")},
        "radius": radius, "centre": centre.tolist(), "png": base64.b64encode(buf.getvalue()).decode(),
    }


def strand_mesh(items, nseg=3):
    """items: (base, dir, normal, length, width, lift, id) -> tapered ribbons, uv = (along, id)."""
    V, UV, F = [], [], []
    for base, d, nn, length, width, lift, sid in items:
        side = np.cross(d, nn)
        side /= np.linalg.norm(side) + 1e-12
        k0 = len(V)
        for k in range(nseg + 1):
            tt = k / nseg
            c = base + d * length * tt + nn * lift * tt * tt
            half = width * (1 - 0.85 * tt) / 2
            V += [c - side * half, c + side * half]
            UV += [[tt, sid], [tt, sid]]
        for k in range(nseg):
            a = k0 + 2 * k
            F += [[a, a + 1, a + 2], [a + 1, a + 3, a + 2]]
    return np.array(V), np.array(UV), np.array(F)



def build_hair_strands(Hp, Th, dh, s_m, count=3800):
    """Short cropped hair: strands that follow the skull from the crown, stopping above the hairline."""
    rng = np.random.default_rng(11)
    Hn = vertex_normals(Hp, Th)
    area = np.linalg.norm(np.cross(Hp[Th[:, 1]] - Hp[Th[:, 0]], Hp[Th[:, 2]] - Hp[Th[:, 0]]), axis=1)
    ok = dh[Th].mean(1) > 0.04
    w = np.where(ok, area, 0)
    tri = rng.choice(len(Th), count, p=w / w.sum())
    b = rng.dirichlet([1, 1, 1], count)
    idx = Th[tri]
    pos = (Hp[idx] * b[..., None]).sum(1)
    nrm = (Hn[idx] * b[..., None]).sum(1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    drop = (dh[idx] * b).sum(1) * s_m
    crown = Hp[np.argmax(Hp[:, 1])]
    g = np.array([0, -1.0, 0])
    items = []
    for p0, n0, dr in zip(pos, nrm, drop):
        v = p0 - crown
        t = v - n0 * (v @ n0)
        t /= np.linalg.norm(t) + 1e-9
        gt = g - n0 * (g @ n0)
        gt /= np.linalg.norm(gt) + 1e-9
        d = 0.55 * t + 0.6 * gt + rng.normal(0, 0.12, 3)
        d -= n0 * (d @ n0) * 0.9
        d /= np.linalg.norm(d)
        L = rng.uniform(0.014, 0.030)
        L = min(L, max(0.004, 1.1 * dr / max(0.25, -d[1])))
        items.append((p0 - n0 * 0.0006, d, n0, L, rng.uniform(0.0010, 0.0016), rng.uniform(0.0008, 0.0035), rng.uniform()))
    V, UV, F = strand_mesh(items, nseg=3)
    print("hair strands:", len(items), "->", len(V), "verts")
    return V, UV, F


def superquad(a, b, c, e1, e2, nu=10, nv=8):
    """A rounded box (superellipsoid) with half-extents a, b, c; returns verts and triangles."""
    sp = lambda x, e: np.sign(x) * np.abs(x) ** e
    V, F = [], []
    for j in range(nv + 1):
        phi = -np.pi / 2 + np.pi * j / nv
        for i in range(nu):
            th = 2 * np.pi * i / nu
            V.append([a * sp(np.cos(phi), e1) * sp(np.cos(th), e2), b * sp(np.sin(phi), e1),
                      c * sp(np.cos(phi), e1) * sp(np.sin(th), e2)])
    for j in range(nv):
        for i in range(nu):
            p0, p1 = j * nu + i, j * nu + (i + 1) % nu
            F += [[p0, p0 + nu, p1], [p1, p0 + nu, p1 + nu]]
    return np.array(V), np.array(F)


def build_mouth(B):
    """Teeth, gums, tongue and a dark cavity behind the lips, in scene metres (rest pose, closed jaw)."""
    my = float(B.scene([0, 6.605, 0])[1])
    zf = 0.1393
    widths = [0.0086, 0.0068, 0.0078, 0.0068, 0.0066]
    ivory = np.array([0.74, 0.69, 0.58])
    gum = np.array([0.62, 0.24, 0.27])

    def arch(upper):
        R = 0.034
        V, C, F = [], [], []
        x = 0.0
        slots = []
        for w in widths:
            slots += [(x + w / 2, w), (-(x + w / 2), w)]
            x += w
        for cx, w in slots:
            z = zf - (0.0 if upper else 0.0022) - cx * cx / (2 * R)
            ang = np.arctan(cx / R)
            h = (0.0105 if abs(cx) < 0.009 else 0.0090 if abs(cx) < 0.0165 else 0.0082) if upper else (0.0072 if abs(cx) < 0.009 else 0.0074)
            e = h * 0.5
            v, f = superquad(w * 0.5, e, 0.0019, 0.42, 0.42)
            # taper the thickness toward the gum line and orient along the arch
            t = (v[:, 1] / e + 1) / 2
            depth = 1 - 0.35 * (t if upper else 1 - t)
            v[:, 2] *= depth
            v[:, 2] += 0.0019
            c, sn = np.cos(ang), np.sin(ang)
            rot = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
            y0 = my - 0.0036 + e if upper else my - 0.0004 - e
            v = v @ rot.T + np.array([cx, y0, z - 0.0019 * c])
            col = ivory[None, :] * (0.88 + 0.12 * (t if not upper else 1 - t))[:, None]
            col = col * np.array([1, 0.98, 0.93]) if upper else col
            V.append(v)
            C.append(col)
            F.append(f + sum(len(q) for q in V[:-1]))
        # gum band along the arch above (upper) or below (lower) the teeth
        n = 26
        xs = np.linspace(-0.0385, 0.0385, n)
        rows = 5
        gv, gc, gf = [], [], []
        for r in range(rows):
            u = r / (rows - 1)
            for xx in xs:
                z = zf - (0.0 if upper else 0.0022) - xx * xx / (2 * R) - 0.0008 + 0.0022 * (1 - abs(u - 0.2)) * 0 + 0.0014 * np.sin(np.pi * min(u * 1.4, 1))
                if upper:
                    y = my - 0.0036 + 0.0105 - 0.0016 + u * 0.0075
                else:
                    y = my - 0.0004 - 0.0074 + 0.0016 - u * 0.0075
                gv.append([xx, y, z])
                gc.append(gum * (1.0 - 0.45 * u))
        for r in range(rows - 1):
            for i in range(n - 1):
                a = r * n + i
                gf += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
        # a shadowed plate just behind the teeth so the gaps between them are not see-through
        pv, pc, pf = [], [], []
        top = my - 0.0036 + 0.0105 if upper else my - 0.0004
        bot = my - 0.0036 if upper else my - 0.0004 - 0.0074
        for ri, yy in enumerate((bot, top)):
            for xx in xs:
                pv.append([xx, yy, zf - (0.0 if upper else 0.0022) - xx * xx / (2 * R) - 0.0022])
                pc.append(ivory * 0.62)
        for i in range(n - 1):
            pf += [[i, i + 1, n + i], [i + 1, n + i + 1, n + i], [i, n + i, i + 1], [i + 1, n + i, n + i + 1]]
        base = sum(len(q) for q in V)
        V.append(np.array(pv))
        C.append(np.array(pc))
        F.append(np.array(pf) + base)
        base = sum(len(q) for q in V)
        V.append(np.array(gv))
        C.append(np.array(gc))
        F.append(np.array(gf) + base)
        return np.concatenate(V), np.concatenate(C), np.concatenate(F)

    uV, uC, uF = arch(True)
    lV, lC, lF = arch(False)

    # tongue: a flattened ellipsoid with a raised tip
    tv, tf = superquad(0.0155, 0.0062, 0.0215, 0.8, 0.8, nu=16, nv=10)
    tv = tv * [1, 1, 1] + np.array([0, my - 0.0072, 0.1185])
    ty = (tv[:, 2] - 0.1185) / 0.0215
    tv[:, 1] += 0.0025 * np.clip(ty, 0, 1) ** 2
    tc = np.tile(np.array([0.66, 0.24, 0.28]), (len(tv), 1)) * (0.8 + 0.2 * np.clip(tv[:, 1] - (my - 0.0134), 0, 0.012)[:, None] / 0.012)
    # cavity: an inverted dark ellipsoid behind the teeth
    cv, cf = superquad(0.036, 0.024, 0.030, 0.9, 0.9, nu=16, nv=10)
    cv = cv + np.array([0, my - 0.0035, 0.1105])
    cf = cf[:, [0, 2, 1]]
    cc = np.tile(np.array([0.20, 0.055, 0.065]), (len(cv), 1))
    q = lambda a: np.round(a * POS_Q).astype("<i2")
    c8 = lambda a: np.round(np.clip(a, 0, 1) * 255).astype("u1")
    print("mouth: teeth/gum", len(uV), "+", len(lV), "verts; tongue", len(tv))
    return {
        "mUVerts": q(uV), "mUCol": c8(uC), "mUFaces": uF.astype("<u2"),
        "mLVerts": q(lV), "mLCol": c8(lC), "mLFaces": lF.astype("<u2"),
        "mTVerts": q(tv), "mTCol": c8(tc), "mTFaces": tf.astype("<u2"),
        "mCVerts": q(cv), "mCCol": c8(cc), "mCFaces": cf.astype("<u2"),
    }


def build_brows(B, T0, P, Nrm):
    """Brow ribbons laid on the face, one per side, as (u along, v across) strips."""
    face = (P[T0][:, :, 2].min(1) > -0.1) & (P[T0][:, :, 1].min(1) > B.scene([0, 6.6, 0])[1])
    tri = P[T0[face]]
    nrm = np.stack([Nrm[T0[face]][:, k] for k in range(3)], 1)

    def surface(x, y):
        """z and unit normal on the front of the face at scene (x, y)."""
        o = np.array([x, y, 1.0])
        d = np.array([0, 0, -1.0])
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        p = np.cross(d, e2)
        det = (e1 * p).sum(1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        tv = o - tri[:, 0]
        u = (tv * p).sum(1) * inv
        q = np.cross(tv, e1)
        v = (d * q).sum(1) * inv
        t = (e2 * q).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
        if not hit.any():
            return None
        i = np.argmin(np.where(hit, t, 9))
        z = 1.0 - t[i]
        w = np.array([1 - u[i] - v[i], u[i], v[i]])
        nn = (nrm[i] * w[:, None]).sum(0)
        return z, nn / np.linalg.norm(nn)

    eye = B.scene(np.array([-0.308, 7.284, 1.245]))
    # brow centre line relative to the right eye centre, in decimetres: (inward x, up y, half width)
    pts = [(0.20, 0.210, 0.017), (0.15, 0.245, 0.025), (0.06, 0.280, 0.031), (-0.06, 0.288, 0.030),
           (-0.17, 0.268, 0.026), (-0.26, 0.232, 0.019), (-0.33, 0.185, 0.011)]
    NS = 6
    from scipy.interpolate import CubicSpline
    arr = np.array(pts)
    tt = np.linspace(0, 1, len(arr))
    cs = CubicSpline(tt, arr)
    ts = np.linspace(0, 1, 28)
    verts, uv, tris = [], [], []
    for i, tk in enumerate(ts):
        x, y, hw = cs(tk)
        px, py = eye[0] + x * B.s, eye[1] + y * B.s
        r = surface(px, py)
        if r is None:
            raise SystemExit(f"brow point missed the face {tk} {px} {py}")
        z, nn = r
        # tangent of the centre line in screen space
        x2, y2, _ = cs(min(1, tk + 0.01))
        x1, y1, _ = cs(max(0, tk - 0.01))
        tg = np.array([(x2 - x1) * B.s, (y2 - y1) * B.s, 0.0])
        tg /= np.linalg.norm(tg)
        side = np.cross(nn, tg)
        side /= np.linalg.norm(side)
        for k in range(NS + 1):
            c = k / NS * 2 - 1
            p = np.array([px, py, z]) + side * c * hw * B.s + nn * 0.0010
            verts.append(p)
            uv.append([tk, (c + 1) / 2])
    for i in range(len(ts) - 1):
        for k in range(NS):
            a = i * (NS + 1) + k
            tris += [[a, a + 1, a + NS + 1], [a + 1, a + NS + 2, a + NS + 1]]
    verts, uv, tris = np.array(verts), np.array(uv), np.array(tris)
    print("brow:", len(verts), "verts")
    rng = np.random.default_rng(7)

    # brow hairs: laid along the brow line, rising on the inner half and drooping at the tail
    items = []
    for _ in range(300):
        tk = rng.uniform(0, 1) ** 1.1
        if tk < 0.08 and rng.uniform() < 0.6:
            continue
        x, y, hw = cs(tk)
        x1, y1, _ = cs(min(1, tk + 0.01))
        x0, y0, _ = cs(max(0, tk - 0.01))
        tg = np.array([x1 - x0, y1 - y0])
        tg /= np.linalg.norm(tg)
        up = np.array([tg[1], -tg[0]])
        c = rng.uniform(-1, 1) * 0.9
        px = eye[0] + (x + up[0] * c * hw * 0.9) * B.s
        py = eye[1] + (y + up[1] * c * hw * 0.9) * B.s
        r = surface(px, py)
        if r is None:
            continue
        z, nn = r
        ang = np.radians(np.interp(tk, [0, 0.4, 1], [72, 24, -34]) + rng.normal(0, 9))
        flat = np.array([tg[0], tg[1], 0.0])
        flat -= nn * (flat @ nn)
        flat /= np.linalg.norm(flat)
        upv = np.array([up[0], up[1], 0.0])
        upv -= nn * (upv @ nn)
        upv /= np.linalg.norm(upv)
        d = np.cos(ang) * flat + np.sin(ang) * upv
        d -= nn * (d @ nn) * 0.7
        d /= np.linalg.norm(d)
        items.append((np.array([px, py, z]) + nn * 0.0006, d, nn, rng.uniform(0.0030, 0.0050) * (0.6 + 0.6 * np.sin(np.pi * min(max(tk, 0.05), 0.98))), 0.00048, 0.0007, rng.uniform()))
    kv, kuv, kf = strand_mesh(items)
    print("brow hairs:", len(items))

    # eyelashes along the lid margin (the same ellipse the skin shader darkens)
    lash = []
    for upper, count in ((True, 34),):
        for i in range(count):
            th = (np.linspace(0.10, 0.88, count)[i] if upper else np.linspace(-0.12, -0.78, count)[i]) * np.pi
            th += rng.normal(0, 0.012)
            px = eye[0] + 0.0172 * np.cos(th)
            py = eye[1] + (0.0070 if upper else 0.0050) * np.sin(th) * (1.0 if upper else 0.8) + (0.0003 if upper else -0.0003)
            r = surface(px, py)
            if r is None:
                continue
            z, nn = r
            radial = np.array([np.cos(th), np.sin(th) * 1.4, 0.0])
            radial /= np.linalg.norm(radial)
            d = radial * 0.55 + nn * (0.75 if upper else 0.45)
            d += np.array([0, 0.35 if upper else -0.1, 0])
            d /= np.linalg.norm(d)
            outer = 1 - abs(np.cos(th)) * 0.0 - (np.cos(th) + 1) / 2 * 0.45
            length = (0.0030 + 0.0030 * np.sin(th if upper else -th) * outer) * (1 if upper else 0.4) * rng.uniform(0.85, 1.15)
            lash.append((np.array([px, py, z]) + nn * 0.0004, d, nn, length, 0.00095, 0.0012 if upper else 0.0003, rng.uniform()))
    lv, luv, lf = strand_mesh(lash)
    print("lashes:", len(lash))
    q = lambda a: np.round(a * POS_Q).astype("<i2")
    u16 = lambda a: np.round(a * 65535).astype("<u2")
    return {
        "arrays": {"rVerts": q(verts), "rUv": u16(uv), "rFaces": tris.astype("<u2"),
                   "kVerts": q(kv), "kUv": u16(kuv), "kFaces": kf.astype("<u2"),
                   "lVerts": q(lv), "lUv": u16(luv), "lFaces": lf.astype("<u2")},
        "meta": {"eye": eye.tolist()},
    }


def main():
    out = build()
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
