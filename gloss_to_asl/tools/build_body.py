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

BONES = ["torso", "neck", "head"] + [f"{p}{s}" for s in "RL" for p in ("up", "f0", "f1", "f2", "wr")]
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
        newV = np.zeros((n + len(ends), 3))
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
            elif base == "shoulder01":
                np.add.at(W[:, arm("up")], idx, w * 0.5)
                np.add.at(W[:, 0], idx, w * 0.5)
            elif base == "neck01":
                np.add.at(W[:, 0], idx, w * 0.5)
                np.add.at(W[:, 1], idx, w * 0.5)
            elif base == "neck02":
                np.add.at(W[:, 1], idx, w)
            elif base == "neck03":
                np.add.at(W[:, 1], idx, w * 0.5)
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
    Nrm = vertex_normals(P, T0)

    # ---- body
    Pb, Tb, Wb = compact(P, T0, Wd)
    used_idx = np.unique(T0)
    print("body:", len(Pb), "verts,", len(Tb), "tris")

    # ---- shirt
    headw = Wd[:, 1] + Wd[:, 2]
    y_dm = V[:, 1]
    neckline = 5.95 - 0.15 * np.clip((V[:, 2] - 0.5) / 0.9, 0, 1)
    shirt_v = (headw < 0.5) & (y_dm > HEM_DM) & (y_dm < neckline + 0.6) & (B.ylocal < CUFF)
    shirt_t = shirt_v[T0].all(1)
    Ts = T0[shirt_t]
    off = np.where(np.abs(V[:, 0]) > 1.7, 0.0032, 0.0048)
    sh = P + Nrm * off[:, None]
    bnd = boundary_edges(Ts)
    strips = []
    for a, b in bnd:
        strips += [[a, b, n + a], [b, n + b, n + a]]
    Pall = np.concatenate([sh, P])
    Ts_all = np.concatenate([Ts, np.array(strips, int).reshape(-1, 3)]) if strips else Ts
    Ps_, Ts_, Ws_ = compact(Pall, Ts_all, np.concatenate([Wd, Wd]))
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
    thick = 0.0010 + 0.0085 * smoothstep(-0.05, 0.55, dh) * (1 + 0.35 * smoothstep(7.8, 8.45, Hd[:, 1]))
    Hp = Hp + Hn * thick[:, None]
    Th = Th[(dh > -0.06)[Th].any(1)]
    Hp, Th, dh = compact(Hp, Th, dh)
    hl = np.round(np.clip(dh, -1.2, 1.2) * 100).astype("i1")
    print("hair:", len(Hp), "verts,", len(Th), "tris")

    # ---- eyes (fitted to the base mesh with the .mhclo anchors)
    eyes = build_eyes(B)

    # ---- joints / landmarks in scene space
    J = lambda name: B.scene(h.head(name))
    right = {"shoulder": J("upperarm01.R"), "elbow": J("lowerarm01.R"), "wrist": J("wrist.R")}
    eye_c = B.scene(np.array([-0.308, 7.284, 1.245]))
    # rest orientation of the hand in the body frame: Vn = R0 (P - wrist)
    brows = build_brows(B, T0, P, Nrm)

    arrays = [
        ("bVerts", np.round(Pb * POS_Q).astype("<i2")), ("bFaces", Tb.astype("<u2")),
    ]
    bi, bw = top4(Wb)
    arrays += [("bSkinIdx", bi.astype("u1")), ("bSkinW", np.round(bw * 255).astype("u1")),
               ("bCover", shirt_v[used_idx].astype("u1"))]
    si, sw = top4(Ws_)
    arrays += [("sVerts", np.round(Ps_ * POS_Q).astype("<i2")), ("sFaces", Ts_.astype("<u2")),
               ("sSkinIdx", si.astype("u1")), ("sSkinW", np.round(sw * 255).astype("u1"))]
    arrays += [("hVerts", np.round(Hp * POS_Q).astype("<i2")), ("hFaces", Th.astype("<u2")), ("hHl", hl)]
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
    pts = [(0.26, 0.205, 0.019), (0.18, 0.245, 0.027), (0.06, 0.280, 0.031), (-0.06, 0.288, 0.030),
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
            p = np.array([px, py, z]) + side * c * hw * B.s + nn * 0.0004
            verts.append(p)
            uv.append([tk, (c + 1) / 2])
    for i in range(len(ts) - 1):
        for k in range(NS):
            a = i * (NS + 1) + k
            tris += [[a, a + 1, a + NS + 1], [a + 1, a + NS + 2, a + NS + 1]]
    verts, uv, tris = np.array(verts), np.array(uv), np.array(tris)
    print("brow:", len(verts), "verts")
    return {
        "arrays": {"rVerts": np.round(verts * POS_Q).astype("<i2"), "rUv": np.round(uv * 65535).astype("<u2"),
                   "rFaces": tris.astype("<u2")},
        "meta": {"eye": eye.tolist()},
    }


def main():
    out = build()
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
