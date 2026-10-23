#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
empilement.py — Empilement optimal
==================================

Peut empiler des objets unitaires dans différents conteneurs :

1. carré dans carré        (défaut)  — n carrés unité dans un carré de côté a
2. cercle dans cercle                — n cercles unité (r = 0.5) dans un cercle de rayon R
3. cercle dans carré                 — n cercles unité (r = 0.5) dans un carré de côté a
4. carré dans cercle                 — n carrés unité dans un cercle de rayon R

L'objectif est de trouver la taille minimale du conteneur (a ou R).
L'IA (Deep-Set + descente + polissage) cherche l'empilement optimal.

Méthode
-------
1. Un agent (réseau de neurones, numpy pur, gradients analytiques vérifiés
   par différences finies --gradcheck) prédit une configuration pour chaque
   objet. Architecture Deep-Set équivariante : tronc partagé, synthèse
   invariante (moyenne), puis tête par objet.
2. La perte est une barrière lisse et différentiable, adaptée au mode :
   - séparation par axe SAT lissé (log-sum-exp) pour des carrés, simple
     distance - 2r pour des cercles ;
   - contenance dans [0, a]² (conteneur carré) ou dans le disque |x| <= a
     (conteneur circulaire), avec les points extrêmes propres à chaque forme.
   Optimisée par Adam. Existe en version numpy de référence et en version
   numba (JIT), équivalentes à la machine.
3. Polissage : raffinement direct des coordonnées, durcissement (tau -> 0),
   puis L-BFGS-B. Complété par une trempe simulée (--anneal-iters) et des
   relances « coup de poing » (--kicks).
4. Amorces variées, adaptées au conteneur : grille, coins pivotés, aléatoire,
   charpentes, bandes diagonales, cadres, rangées + bloc pivoté, palette
   discrète d'angles (conteneur carré) ; grilles, anneaux concentriques et
   tirages dans le disque (conteneur circulaire).
   Chaque membre est évalué exactement ; les meilleurs sont polis.
   Pour n = k²+1 (k >= 2), une construction de Göbel généralisée par récurrence
   fournit un certificat exact au côté k + √2/2.
5. Recherche de C(n) : descente géométrique de `a` suivie d'une bissection.
   Chaque résultat est CERTIFIÉ rigoureusement :
   - snapping (snap_config) : ajustement des contacts actifs ;
   - dilatation (rescale_to_feasible) : empilement prouvé valide à a' >= a ;
   - resserrement (tighten_a) : conteneur au plus juste du support réel.
   La réponse est donc un MAJORANT RIGOUREUX de C(n).

Parallélisme : agents, polissages et trempe sont répartis sur un pool de
processus (--workers). --spec K lance K essais spéculatifs pendant la
descente ; --runs K lance K runs multi-graines successifs et retient le meilleur.

RÈGLES D'ENREGISTREMENT (solution_acceptable) — une solution refusée n'est
NI enregistrée NI générée :
  1. solution SUR GRILLE : côté `a` entier (n = k², k²-1, ...) -> rien ;
  2. cas n = k²+1 : borne maximale k + √2/2 (ex. n = 37 -> 6.707107) ;
     toute solution au-dessus est refusée.

Sortie
------
    <outdir>/figures/empilement_n<n>_<mode>.png   figure individuelle

Usage
-----
    python3 empilement.py -n 11 --pack 1           # carré dans carré
    python3 empilement.py -n 5  --pack 2           # cercle dans cercle
    python3 empilement.py -n 7  --pack 3           # cercle dans carré
    python3 empilement.py -n 9  --pack 4           # carré dans cercle
    python3 empilement.py -n 11 --iters 500        # plus d'effort
    python3 empilement.py -n 17 --runs 4           # 4 runs multi-graines
    python3 empilement.py -n 26 --time-budget 20   # 20 s pour ce n
    python3 empilement.py --range '10;50'          # n = 10 à 50
    python3 empilement.py --gradcheck
"""

import argparse
import atexit
import math
import multiprocessing
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# matplotlib en mode non interactif (fichiers seulement)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

SQRT2 = math.sqrt(2.0)
HALF_SQRT2 = 0.5 * SQRT2
HALF_PI = 0.5 * math.pi
EPS = 1e-12
FEAS_TOL = 1e-7
# offsets des 4 sommets d'un carré unité (sens direct), dans le repère local
OFF = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
# points extrêmes d'un cercle unité (rayon r) : suffisent pour la boîte [0,a]²
CARD = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

# Références externes supprimées : seul le calcul de borne supérieure est utilisé.



def bound_k2_plus1(n):
    """Pour n = k² + 1, borne maximale autorisée : k + √2/2.
    (Ex. n = 37 = 6²+1 : borne 6 + √2/2 = 6.707107.)"""
    k = math.isqrt(n)
    if k * k + 1 == n and k >= 1:
        return k + math.sqrt(2.0) / 2.0
    return None


def solution_acceptable(n, a, tol=1e-6):
    """Règles d'enregistrement — n'ont de sens que pour carré dans carré.
    Sinon on autorise (pas de tables/borne appliquées)."""
    c = _ctx()
    if not (c['sqs'] and c['sqc']):
        return True, ""
    r = round(a)
    if abs(a - r) <= tol and r >= 1:
        return False, (f"solution sur grille (a = {r:g}, entier — n = "
                       f"{r}²±) : rien ni enregistré ni généré")
    b = bound_k2_plus1(n)
    if b is not None and a > b + 1e-9:
        k = math.isqrt(n)
        return False, (f"n = {k}²+1 : a = {a:.6f} > borne maximale "
                       f"{b:.6f} (√{k}² + √2/2) : rien ni enregistré "
                       f"ni généré")
    return True, ""



PACK = "sq_in_sq"
# Drapeaux de géométrie, mis à jour par set_pack() : utilisés dans les
# fonctions chaude (perte, violation, JIT) où _ctx() serait trop coûteux.
_CIRC_SHAPE = False        # les objets sont des cercles (sinon des carrés)
_CIRC_CONT = False         # le conteneur est un cercle (sinon un carré)
_R = 0.5                   # rayon d'un cercle unité


def set_pack(p):
    global PACK, _CIRC_SHAPE, _CIRC_CONT, _R
    PACK = p
    _CIRC_SHAPE = p in ("circ_in_circ", "circ_in_sq")
    _CIRC_CONT = p in ("circ_in_circ", "sq_in_circ")
    _R = 0.5
    return PACK


def _ctx():
    return dict(p=PACK, sqc=not _CIRC_CONT, cic=_CIRC_CONT,
                sqs=not _CIRC_SHAPE, cis=_CIRC_SHAPE, r=_R)


def area_lower_bound(n):
    """Borne inférieure d'aire : a² * aire(conteneur) >= n * aire(objet).

    Carré dans carré : √n ; cercle dans carré : √(nπ/4) ; cercle dans cercle :
    √n/2 ; carré dans cercle : √(n/π)."""
    obj = 1.0 if not _CIRC_SHAPE else math.pi * _R * _R
    cont = 1.0 if not _CIRC_CONT else math.pi
    return math.sqrt(n * obj / cont)


def density(n, a):
    """Densité (fraction d'aire du conteneur occupée par les objets)."""
    obj = 1.0 if not _CIRC_SHAPE else math.pi * _R * _R
    cont = a * a if not _CIRC_CONT else math.pi * a * a
    return n * obj / cont


def initial_upper_bound(n):
    """Taille de conteneur pour laquelle la grille de ceil(√n) objets tient.

    Carré : côté ceil(√n) (grille exacte). Cercle : rayon de la grille
    correspondante, dilatée jusqu'à contenir son point le plus extrême."""
    k = max(1, int(math.ceil(math.sqrt(n))))
    if not _CIRC_CONT:
        return float(k)
    extent = 0.5 * SQRT2 if not _CIRC_SHAPE else _R
    return (k - 1) * 0.5 * SQRT2 + extent


# ===========================================================================
#  Sortie : uniquement figures/empilement_n<n>_<mode>.png
# ===========================================================================

DEFAULT_OUTDIR = os.environ.get("EMPILEMENT_OUTDIR", "resultats")


def dir_figures(outdir=DEFAULT_OUTDIR):
    return os.path.join(outdir, "figures")


def figure_path(n, outdir=DEFAULT_OUTDIR):
    try:
        p = PACK
    except Exception:
        p = "sq_in_sq"
    return os.path.join(dir_figures(outdir), f"empilement_n{n}_{p}.png")


_POOL = None
_IN_WORKER = False


def setup_pool(workers):
    """Crée (workers > 1) ou détruit (workers <= 1) le pool de processus."""
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None
    if workers and int(workers) > 1:
        _POOL = multiprocessing.get_context("fork").Pool(
            processes=int(workers))
    return _POOL


atexit.register(lambda: setup_pool(1))


def _pmap(fn, tasks):
    """map parallèle si le pool existe, sinon séquentiel (repli sûr).

    Séquentiel aussi quand _IN_WORKER (worker B3/B4) : un worker ne doit pas
    réutiliser le pool (deadlock / sur-embauche) — voir plus haut."""
    tasks = list(tasks)
    if _POOL is None or _IN_WORKER or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    return _POOL.map(fn, tasks, chunksize=1)


# ===========================================================================
#  Géométrie : fonctions de support, perte lisse + gradients analytiques
# ===========================================================================

def support(alpha):
    """Rayon de support (demi-largeur) d'un carré unité tourné de alpha."""
    return 0.5 * (np.abs(np.cos(alpha)) + np.abs(np.sin(alpha)))


def support_grad(alpha):
    """d(support)/d(alpha)."""
    c, s = np.cos(alpha), np.sin(alpha)
    return 0.5 * (-np.sign(c) * s + np.sign(s) * c)


def build_geometry(centers, thetas, a):
    """Géométrie unifiée (carrés SAT 4-axes / cercles distance)."""
    c = _ctx(); n = centers.shape[0]
    d = centers[None, :, :] - centers[:, None, :]
    ti, tj = thetas[:, None], thetas[None, :]
    ct, st = np.cos(thetas), np.sin(thetas)
    R = np.stack([np.stack([ct, -st], -1), np.stack([st, ct], -1)], -2)
    if c['sqs']:
        phi = np.stack([ti + 0*tj, ti + HALF_PI + 0*tj, tj + 0*ti, tj + HALF_PI + 0*ti], axis=-1)
        u = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
        u_perp = np.stack([-np.sin(phi), np.cos(phi)], axis=-1)
        p = (d[:, :, None, :] * u).sum(axis=-1)
        ai, aj = ti[:, :, None] - phi, tj[:, :, None] - phi
        hi, hj = support(ai), support(aj)
        g = np.abs(p) - hi - hj
        V = centers[:, None, :] + OFF @ R.transpose(0, 2, 1)
    else:
        dist = np.sqrt(np.maximum((d ** 2).sum(axis=2), EPS))
        g = (dist - 2.0 * c['r'])[:, :, None]
        phi = np.zeros((n, n, 1)); u = np.zeros((n, n, 1, 2)); u_perp = np.zeros((n, n, 1, 2))
        p = dist[:, :, None]; ai = aj = np.zeros((n, n, 1)); hi = hj = np.zeros((n, n, 1))
        # points extrêmes d'un cercle : centre + r * directions cardinales
        V = centers[:, None, :] + c['r'] * CARD[None, :, :]
    return dict(n=n, d=d, phi=phi, u=u, u_perp=u_perp, p=p, ai=ai, aj=aj,
                hi=hi, hj=hj, g=g, off=OFF, R=R, V=V, a=a)


def _loss_and_backward_np(centers, thetas, a, tau, lam_p=1.0, lam_v=1.0):
    """Perte lisse GÉNÉRIQUE (4 modes) et gradient analytique complet.

    L = lam_p * moyenne_pairs( relu(-sep_tau)^2 )
      + lam_v * moyenne_bord( relu(-v)^2 + relu(v-a)^2 )        (carré)
        lam_v * moyenne_bord( relu(|v|-a)^2 )                   (cercle)

    - carrés : la séparation est le max lissé (log-sum-exp) des gaps sur les
      4 axes de séparation (théorème de l'axe séparateur) ;
    - cercles : la séparation est simplement |cj - ci| - 2r ;
    - conteneur carré : les points extrêmes (sommets ou points cardinaux)
      doivent rester dans [0, a]² ;
    - conteneur cercle : les sommets des carrés (ou le centre des cercles,
      décalé de r) doivent rester dans le disque |x| <= a.

    Version numpy pur — conservée comme référence pour le test d'équivalence
    avec la version JIT ; loss_and_backward pointe dessus sans numba.
    """
    c = _ctx()
    n = centers.shape[0]
    geo = build_geometry(centers, thetas, a)
    V = geo["V"]
    dcenters = np.zeros_like(centers)
    dthetas = np.zeros_like(thetas)
    L_pairs = 0.0

    # ------------------------------- paires -------------------------------
    if n > 1:
        iu = np.triu_indices(n, k=1)
        d = geo["d"]
        if not c['cis']:
            g, p, u, u_perp = geo["g"], geo["p"], geo["u"], geo["u_perp"]
            ai, aj = geo["ai"], geo["aj"]
            gg = g[iu]
            m = gg.max(axis=1, keepdims=True)
            sep = m + tau * np.log(np.exp((gg - m) / tau).sum(axis=1,
                                                            keepdims=True))
            L_pairs = (np.maximum(-sep, 0.0) ** 2).mean()
            dsep = -2.0 * np.maximum(-sep, 0.0)                 # dpen/dsep
            w = np.exp((gg - m) / tau)
            w /= w.sum(axis=1, keepdims=True)                   # softmax
            dgg = (dsep * w) * (lam_p / gg.shape[0])
            dg = np.zeros_like(g)
            dg[iu] = dgg
            s = np.sign(p)
            Hpi, Hpj = support_grad(ai), support_grad(aj)
            dphi = s * (d[:, :, None, :] * u_perp).sum(axis=-1) + Hpi + Hpj
            coef = dg
            Dd = ((coef * s)[..., None] * u).sum(axis=2)
            dcenters += (Dd.transpose(1, 0, 2) - Dd).sum(axis=1)
            dep_i = np.zeros_like(coef); dep_i[..., 0:2] = 1.0
            dep_j = np.zeros_like(coef); dep_j[..., 2:4] = 1.0
            A = coef * dphi * dep_i - coef * Hpi
            B = coef * dphi * dep_j - coef * Hpj
            dthetas += A.sum(axis=(1, 2)) + B.sum(axis=(0, 2))
        else:
            dist = np.sqrt(np.maximum((d ** 2).sum(axis=2), EPS))
            g = dist[iu] - 2.0 * c['r']
            L_pairs = (np.maximum(-g, 0.0) ** 2).mean()
            dg = -(2.0 * np.maximum(-g, 0.0)) * (lam_p / g.shape[0])
            ii, jj = iu
            inv = np.zeros((n, n))
            inv[ii, jj] = dg / np.maximum(dist[ii, jj], EPS)
            dd = inv[:, :, None] * d
            # dg/dcj = +d/|d| ; dg/dci = -d/|d|
            dcenters += dd.sum(axis=0) - dd.sum(axis=1)

    # ----------------------------- contenance -----------------------------
    if c['sqc']:
        pen_v = np.maximum(-V, 0.0) ** 2 + np.maximum(V - a, 0.0) ** 2
        L_vert = pen_v.mean()
        dV = (-2.0 * np.maximum(-V, 0.0) + 2.0 * np.maximum(V - a, 0.0))
        dV *= lam_v / V.size
        dcenters += dV.sum(axis=1)
        if not c['cis']:
            ct2 = np.cos(thetas + 0.5 * math.pi)
            st2 = np.sin(thetas + 0.5 * math.pi)
            R2 = np.stack([np.stack([ct2, -st2], -1),
                           np.stack([st2, ct2], -1)], -2)
            dthetas += (dV * (OFF @ R2.transpose(0, 2, 1))).sum(axis=(1, 2))
    elif not c['cis']:
        # sommets de carrés dans le disque de rayon a
        nv = np.sqrt(np.maximum((V ** 2).sum(axis=2), EPS))
        exc = np.maximum(nv - a, 0.0)
        denom = lam_v / (n * 8.0)
        L_vert = (exc ** 2).sum() * denom
        coef = (2.0 * exc / np.maximum(nv, EPS)) * denom
        dV = coef[..., None] * V
        dcenters += dV.sum(axis=1)
        ct2 = np.cos(thetas + 0.5 * math.pi)
        st2 = np.sin(thetas + 0.5 * math.pi)
        R2 = np.stack([np.stack([ct2, -st2], -1),
                       np.stack([st2, ct2], -1)], -2)
        dthetas += (dV * (OFF @ R2.transpose(0, 2, 1))).sum(axis=(1, 2))
    else:
        # centres de cercles dans le disque de rayon a : |c| + r <= a
        nc = np.sqrt(np.maximum((centers ** 2).sum(axis=1), EPS))
        exc = np.maximum(nc + c['r'] - a, 0.0)
        denom = lam_v / (2.0 * n)
        L_vert = (exc ** 2).sum() * denom
        coef = (2.0 * exc / np.maximum(nc, EPS)) * denom
        dcenters += coef[:, None] * centers

    L = lam_p * L_pairs + L_vert
    return L, dcenters, dthetas, None


def loss_and_backward(centers, thetas, a, tau, lam_p=1.0, lam_v=1.0):
    """Perte lisse + gradient analytique (par défaut : chemin numpy pur).

    Surchargé par la version JIT numba ci-dessous si numba est présent.
    """
    return _loss_and_backward_np(centers, thetas, a, tau, lam_p, lam_v)


# --- version JIT (numba) de loss_and_backward : mêmes maths, boucles natives.
# C'est LA fonction saturante de l'entraînement (gradient analytique à chaque
# itération d'agent, de polissage et de trempe) ; elle évite les temporaires
# (n, n, 4) du chemin numpy. Équivalente à la version numpy à ~1e-14 près
# (vérifié par --gradcheck) ; repli
# numpy automatique si numba est absent.
try:
    import numba as _numba

    @_numba.njit(cache=True)
    def _sg_nb(x):
        """sign numpy : surtout sign(0) = 0 (dérivées de |p| et du support)."""
        if x > 0.0:
            return 1.0
        if x < 0.0:
            return -1.0
        return 0.0

    @_numba.njit(cache=True)
    def _support_nb(al):
        return 0.5 * (abs(math.cos(al)) + abs(math.sin(al)))

    @_numba.njit(cache=True)
    def _support_grad_nb(al):
        c = math.cos(al)
        s = math.sin(al)
        return 0.5 * (-_sg_nb(c) * s + _sg_nb(s) * c)

    @_numba.njit(cache=True)
    def _loss_and_backward_nb(c, th, a, tau, lam_p, lam_v, cs, cc, r):
        n = c.shape[0]
        L = 0.0
        dc = np.zeros_like(c)
        dth = np.zeros_like(th)

        # ------------------------- partie paires (i < j) -------------------
        npairs = n * (n - 1) // 2
        if npairs > 0:
            for i in range(n):
                ti = th[i]
                cix, ciy = c[i, 0], c[i, 1]
                for j in range(i + 1, n):
                    tj = th[j]
                    dx, dy = c[j, 0] - cix, c[j, 1] - ciy
                    if cs == 0:
                        # --- carrés : gaps des 4 axes + max lissé
                        g = np.empty(4)
                        m = -1.0e300
                        for k in range(4):
                            if k == 0:
                                phi = ti
                            elif k == 1:
                                phi = ti + HALF_PI
                            elif k == 2:
                                phi = tj
                            else:
                                phi = tj + HALF_PI
                            ux, uy = math.cos(phi), math.sin(phi)
                            p = dx * ux + dy * uy
                            hi = _support_nb(ti - phi)
                            hj = _support_nb(tj - phi)
                            gk = abs(p) - hi - hj
                            g[k] = gk
                            if gk > m:
                                m = gk
                        se = 0.0
                        for k in range(4):
                            se += math.exp((g[k] - m) / tau)
                        sep = m + tau * math.log(se)
                        relu = -sep if sep < 0.0 else 0.0
                        L += lam_p * relu * relu / npairs
                        dsep = -2.0 * relu
                        wsum = 0.0
                        w = np.empty(4)
                        for k in range(4):
                            w[k] = math.exp((g[k] - m) / tau)
                            wsum += w[k]
                        for k in range(4):
                            ww = w[k] / wsum
                            coef = dsep * ww * (lam_p / npairs)
                            if k == 0:
                                phi = ti
                            elif k == 1:
                                phi = ti + HALF_PI
                            elif k == 2:
                                phi = tj
                            else:
                                phi = tj + HALF_PI
                            ux, uy = math.cos(phi), math.sin(phi)
                            p = dx * ux + dy * uy
                            s = _sg_nb(p)
                            Hpi = _support_grad_nb(ti - phi)
                            Hpj = _support_grad_nb(tj - phi)
                            dphi = (s * (dx * -math.sin(phi)
                                         + dy * math.cos(phi)) + Hpi + Hpj)
                            term = coef * s
                            dc[i, 0] -= term * ux
                            dc[i, 1] -= term * uy
                            dc[j, 0] += term * ux
                            dc[j, 1] += term * uy
                            if k < 2:
                                dth[i] += coef * dphi - coef * Hpi
                                dth[j] += -coef * Hpj
                            else:
                                dth[i] += -coef * Hpi
                                dth[j] += coef * dphi - coef * Hpj
                    else:
                        # --- cercles : gap = distance - 2r
                        dist2 = dx * dx + dy * dy
                        dist = math.sqrt(dist2) if dist2 > 1e-300 else 1e-150
                        gsep = dist - 2.0 * r
                        relu = -gsep if gsep < 0.0 else 0.0
                        L += lam_p * relu * relu / npairs
                        dg = -2.0 * relu * (lam_p / npairs)
                        inv = dg / dist
                        dc[i, 0] -= inv * dx
                        dc[i, 1] -= inv * dy
                        dc[j, 0] += inv * dx
                        dc[j, 1] += inv * dy

        # --------------------------- contenance ----------------------------
        if cc == 0:
            # conteneur carré [0, a]²
            denom = lam_v / (n * 8.0)
            if cs == 0:
                for i in range(n):
                    ti = th[i]
                    ct, st = math.cos(ti), math.sin(ti)
                    for q in range(4):
                        qx, qy = OFF[q, 0], OFF[q, 1]
                        vx = c[i, 0] + qx * ct - qy * st
                        vy = c[i, 1] + qx * st + qy * ct
                        gvx = 0.0
                        gvy = 0.0
                        if vx < 0.0:
                            L += lam_v * vx * vx / (n * 8.0)
                            gvx += 2.0 * vx
                        elif vx > a:
                            L += lam_v * (vx - a) * (vx - a) / (n * 8.0)
                            gvx += 2.0 * (vx - a)
                        if vy < 0.0:
                            L += lam_v * vy * vy / (n * 8.0)
                            gvy += 2.0 * vy
                        elif vy > a:
                            L += lam_v * (vy - a) * (vy - a) / (n * 8.0)
                            gvy += 2.0 * (vy - a)
                        gvx *= denom
                        gvy *= denom
                        dc[i, 0] += gvx
                        dc[i, 1] += gvy
                        dvx = -qx * st - qy * ct
                        dvy = qx * ct - qy * st
                        dth[i] += gvx * dvx + gvy * dvy
            else:
                # cercles : points cardinaux (pas de dépendance à theta)
                for i in range(n):
                    for q in range(4):
                        qx, qy = r * CARD[q, 0], r * CARD[q, 1]
                        vx = c[i, 0] + qx
                        vy = c[i, 1] + qy
                        if vx < 0.0:
                            L += lam_v * vx * vx / (n * 8.0)
                            dc[i, 0] += 2.0 * vx * denom
                        elif vx > a:
                            L += lam_v * (vx - a) * (vx - a) / (n * 8.0)
                            dc[i, 0] += 2.0 * (vx - a) * denom
                        if vy < 0.0:
                            L += lam_v * vy * vy / (n * 8.0)
                            dc[i, 1] += 2.0 * vy * denom
                        elif vy > a:
                            L += lam_v * (vy - a) * (vy - a) / (n * 8.0)
                            dc[i, 1] += 2.0 * (vy - a) * denom
        else:
            # conteneur disque de rayon a, centré en 0
            if cs == 0:
                # carrés : chaque sommet doit rester dans le disque
                denom = lam_v / (n * 8.0)
                for i in range(n):
                    ti = th[i]
                    ct, st = math.cos(ti), math.sin(ti)
                    for q in range(4):
                        qx, qy = OFF[q, 0], OFF[q, 1]
                        vx = c[i, 0] + qx * ct - qy * st
                        vy = c[i, 1] + qx * st + qy * ct
                        nv2 = vx * vx + vy * vy
                        nv = math.sqrt(nv2) if nv2 > 1e-300 else 1e-150
                        ex = nv - a
                        if ex > 0.0:
                            L += lam_v * ex * ex / (n * 8.0)
                            coef = (2.0 * ex / nv) * denom
                            gvx = coef * vx
                            gvy = coef * vy
                            dc[i, 0] += gvx
                            dc[i, 1] += gvy
                            dvx = -qx * st - qy * ct
                            dvy = qx * ct - qy * st
                            dth[i] += gvx * dvx + gvy * dvy
            else:
                # cercles : |centre| + r <= a
                denom = lam_v / (2.0 * n)
                for i in range(n):
                    cx, cy = c[i, 0], c[i, 1]
                    nv2 = cx * cx + cy * cy
                    nv = math.sqrt(nv2) if nv2 > 1e-300 else 1e-150
                    ex = nv + r - a
                    if ex > 0.0:
                        L += lam_v * ex * ex / (2.0 * n)
                        coef = (2.0 * ex / nv) * denom
                        dc[i, 0] += coef * cx
                        dc[i, 1] += coef * cy
        return L, dc, dth

    def loss_and_backward(centers, thetas, a, tau, lam_p=1.0, lam_v=1.0):
        """Perte lisse + gradient analytique (version JIT numba).

        Identique au chemin numpy ci-dessus (~1e-14 près) ; le 4ᵉ élément
        (geo) vaut None sur ce chemin — tous les appels l'ignorent.
        """
        L, dc, dth = _loss_and_backward_nb(
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(thetas, dtype=np.float64),
            float(a), float(tau), float(lam_p), float(lam_v),
            int(_CIRC_SHAPE), int(_CIRC_CONT), float(_R))
        return float(L), dc, dth, None

except ImportError:                                  # numba absent : version numpy
    pass


def exact_violation_np(centers, thetas, a):
    """Violations EXACTES (référence numpy, sans lissage)."""
    c = _ctx(); n = centers.shape[0]; pen = 0.0; r = c['r']
    if c['cis']:
        if n > 1:
            for i in range(n):
                for j in range(i+1,n):
                    d = np.hypot(centers[j,0]-centers[i,0], centers[j,1]-centers[i,1])
                    pen = max(pen, max(0.0, 2.0*r - d))
        if c['cic']:
            R = float(a)
            for i in range(n):
                d = np.hypot(centers[i,0], centers[i,1])
                pen = max(pen, max(0.0, d + r - R))
        else:
            for i in range(n):
                for k in range(2):
                    pen = max(pen, max(0.0, r - centers[i,k]), max(0.0, centers[i,k] + r - float(a)))
        return float(pen)
    if c['sqs']:
        geo = build_geometry(centers, thetas, a)
        if n > 1:
            iu = np.triu_indices(n, k=1)
            g = geo["g"]
            pen_pair = np.maximum(-g[iu].max(axis=1), 0.0).max() if (len(iu[0])>0 and g.shape[-1]>=1) else 0.0
        else:
            pen_pair = 0.0
        if c['sqc']:
            V = geo["V"]
            pen_bord = max(np.maximum(-V, 0.0).max(), np.maximum(V - a, 0.0).max())
        else:
            R = float(a); ct = np.cos(thetas); st = np.sin(thetas); pen_bord = 0.0
            for i in range(n):
                for qx,qy in [(-0.5,-0.5),(0.5,-0.5),(0.5,0.5),(-0.5,0.5)]:
                    vx = centers[i,0] + qx*ct[i] - qy*st[i]
                    vy = centers[i,1] + qx*st[i] + qy*ct[i]
                    pen_bord = max(pen_bord, max(0.0, np.hypot(vx,vy) - R))
        return float(max(pen_pair if n>1 else 0.0, pen_bord))
    return 0.0


# référence numpy accessible même quand la version JIT prend le relais
exact_violation = exact_violation_np


# --- version JIT (numba) de exact_violation : même maths, boucles natives.
# C'est LA fonction saturante (appelée à chaque itération d'entraînement et de
# polissage) ; la compilation la rend ~10-50x plus rapide que le tableau numpy
# temporaires. Identique à la version numpy (testé à 1e-15 près).
try:
    import numba as _numba

    @_numba.njit(cache=True)
    def _exact_violation_nb(c, th, a, cs, cc, r):
        n = c.shape[0]
        pen = 0.0
        HALF = 1.5707963267948966
        # ---- chevauchements ----
        if cs == 0:
            # carrés : axe séparateur, 4 axes par paire
            for i in range(n):
                ti = th[i]
                cix, ciy = c[i, 0], c[i, 1]
                for j in range(i + 1, n):
                    tj = th[j]
                    dx, dy = c[j, 0] - cix, c[j, 1] - ciy
                    best = -1e300
                    for k in range(4):
                        if k == 0:
                            phi = ti
                        elif k == 1:
                            phi = ti + HALF
                        elif k == 2:
                            phi = tj
                        else:
                            phi = tj + HALF
                        ux, uy = math.cos(phi), math.sin(phi)
                        p = dx * ux + dy * uy
                        ai, aj = ti - phi, tj - phi
                        hi = 0.5 * (abs(math.cos(ai)) + abs(math.sin(ai)))
                        hj = 0.5 * (abs(math.cos(aj)) + abs(math.sin(aj)))
                        g = abs(p) - hi - hj
                        if g > best:
                            best = g
                    if best < 0.0 and -best > pen:
                        pen = -best
        else:
            # cercles : distance - 2r
            for i in range(n):
                for j in range(i + 1, n):
                    dx = c[j, 0] - c[i, 0]
                    dy = c[j, 1] - c[i, 1]
                    d = math.sqrt(dx * dx + dy * dy)
                    g = d - 2.0 * r
                    if g < 0.0 and -g > pen:
                        pen = -g
        # ---- contenance ----
        if cc == 0:
            # conteneur carré [0, a]²
            if cs == 0:
                for i in range(n):
                    ct, st = math.cos(th[i]), math.sin(th[i])
                    for qx in (-0.5, 0.5):
                        for qy in (-0.5, 0.5):
                            vx = c[i, 0] + qx * ct - qy * st
                            vy = c[i, 1] + qx * st + qy * ct
                            if -vx > pen:
                                pen = -vx
                            if vx - a > pen:
                                pen = vx - a
                            if -vy > pen:
                                pen = -vy
                            if vy - a > pen:
                                pen = vy - a
            else:
                for i in range(n):
                    for k in range(2):
                        if r - c[i, k] > pen:
                            pen = r - c[i, k]
                        if c[i, k] + r - a > pen:
                            pen = c[i, k] + r - a
        else:
            # conteneur disque de rayon a, centré en 0
            if cs == 0:
                for i in range(n):
                    ct, st = math.cos(th[i]), math.sin(th[i])
                    for qx in (-0.5, 0.5):
                        for qy in (-0.5, 0.5):
                            vx = c[i, 0] + qx * ct - qy * st
                            vy = c[i, 1] + qx * st + qy * ct
                            g = math.sqrt(vx * vx + vy * vy) - a
                            if g > pen:
                                pen = g
            else:
                for i in range(n):
                    d = math.sqrt(c[i, 0] * c[i, 0] + c[i, 1] * c[i, 1])
                    g = d + r - a
                    if g > pen:
                        pen = g
        return pen

    def exact_violation(centers, thetas, a):
        """Violations EXACTES (version JIT numba — voir plus haut pour le doc)."""
        return float(_exact_violation_nb(
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(thetas, dtype=np.float64), float(a),
            int(_CIRC_SHAPE), int(_CIRC_CONT), float(_R)))

except ImportError:                                  # numba absent : version numpy
    pass


def warmup_jit():
    """Compile les fonctions JIT (numba) et importe scipy dans le PARENT.

    À appeler AVANT setup_pool() : avec la méthode « fork », un pool créé
    avant compilation forcerait chaque worker à compiler de son côté (et à
    se disputer le fichier de cache numba). Pré-importe aussi scipy.optimize
    (B5, L-BFGS-B) pour que les workers forkés héritent du code déjà prêt.
    ~0,5 s une fois pour toutes.
    """
    c = np.array([[0.4, 0.6], [1.5, 0.5], [0.9, 1.7]])
    th = np.array([0.0, 0.2, 0.78])
    exact_violation(c, th, 2.0)
    loss_and_backward(c, th, 2.0, 1e-3)
    loss_and_backward(c, th, 2.0, 1e-9)
    try:
        import scipy.optimize          # B5 : L-BFGS-B préchargé avant fork
    except ImportError:
        pass


# ===========================================================================
#  L'agent : un petit réseau de neurones (numpy)
# ===========================================================================

N_FEATURES = 5          # [n/64, √n/8, a/8, (a-borne_aire)/8, 1]
BASE_FEATURES = 3       # config de base : x/8, y/8, theta/8 (par objet)


def init_agent(rng, max_slots, de, H, arch="set"):
    """Poids de l'agent — Deep-Set ÉQUIVARIANT : un tronc partagé par carré
    φ(x_i), une synthèse INVARIANTE g = moyenne_i φ(x_i), puis une tête
    ρ(x_i, g) par carré. Permuter les carrés permute la sortie — la bonne
    invariance pour un problème sans ordre, et une meilleure généralisation
    entre n différents. (Gradient de g : chaque carré reçoit dG/n.)
    """
    nin = de + N_FEATURES + BASE_FEATURES
    H2 = H
    return {
        "E":   rng.normal(0.0, 0.5, (max_slots, de)),
        "W1":  rng.normal(0.0, 1.0, (nin, H)) / math.sqrt(nin),  # tronc φ
        "b1":  np.zeros(H),
        "W2a": rng.normal(0.0, 1.0, (2 * H, H2)) / math.sqrt(2 * H),  # tête ρ
        "b2":  np.zeros(H2),
        "W3":  rng.normal(0.0, 0.1, (H2, 3)) / math.sqrt(H2),     # ≈ résidu nul
        "b3":  np.zeros(3),
    }


def build_inputs(params, n, a, base):
    """Entrées par carré : embedding (appris) + config de base + feats."""
    feat = np.array([n / 64.0, math.sqrt(n) / 8.0, a / 8.0,
                     (a - area_lower_bound(n)) / 8.0, 1.0])
    F = np.broadcast_to(feat, (n, N_FEATURES))
    # config de base : positions /8 et angles /8 (mêmes ordres de grandeur)
    B = np.concatenate([base[0] / 8.0, base[1][:, None] / 8.0], axis=1)
    return np.concatenate([params["E"][:n], F, B], axis=1)     # (n, de+5+3)


def set_forward(params, X):
    """A4 — Deep-Set : O_i = ρ(x_i, g) avec g = moyenne_j φ(x_j) invariant."""
    Hh = np.tanh(X @ params["W1"] + params["b1"])          # φ(x_i)
    G = Hh.mean(axis=0)                                    # invariant
    Oin = np.concatenate([Hh, np.broadcast_to(G, Hh.shape)], axis=1)
    H2 = np.tanh(Oin @ params["W2a"] + params["b2"])       # ρ partie interne
    O = H2 @ params["W3"] + params["b3"]
    return O, (X, Hh, Oin, H2)


def set_backward(params, cache, dO):
    """Rétropropagation de set_forward.

    Point clé : G = mean(H1) donc dG circule vers CHAQUE carré via dG/n
    (la sortie du carré i dépend des entrées de tous — c'est ça, la couche
    de synthèse du Deep-Set). Gradcheck vérifie cette règle à 1e-6 près.
    """
    X, Hh, Oin, H2 = cache
    nn = Hh.shape[0]
    dW3 = H2.T @ dO
    db3 = dO.sum(axis=0)
    dH2 = dO @ params["W3"].T
    dZ2 = dH2 * (1.0 - H2 * H2)
    dW2a = Oin.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dOin = dZ2 @ params["W2a"].T
    dH1 = dOin[:, :Hh.shape[1]].copy()
    dG = dOin[:, Hh.shape[1]:].sum(axis=0)                 # somme, pas moyenne
    dH1 += dG[None, :] / nn                                # d(mean)/d(H1) = 1/n
    dZ = dH1 * (1.0 - Hh * Hh)
    dW1 = X.T @ dZ
    db1 = dZ.sum(axis=0)
    dX = dZ @ params["W1"].T
    return {"W1": dW1, "b1": db1, "W2a": dW2a, "b2": db2,
            "W3": dW3, "b3": db3}, dX


def agent_config(params, X, base, rot):
    """Configuration prédite par le réseau (résidu autour de `base`)."""
    O, cache = set_forward(params, X)
    centers = base[0] + O[:, :2]
    thetas = base[1] + (O[:, 2] if rot else 0.0 * O[:, 2])
    return centers, thetas, cache, O


# ------------------------------- Adam --------------------------------------

def ravel_params(p):
    return np.concatenate([p[k].ravel() for k in sorted(p)])


def unravel_params(vec, template):
    out, i = {}, 0
    for k in sorted(template):
        sz = template[k].size
        out[k] = vec[i:i + sz].reshape(template[k].shape)
        i += sz
    return out


class Adam:
    def __init__(self, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, vec, grad):
        if self.m is None:
            self.m = np.zeros_like(vec)
            self.v = np.zeros_like(vec)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return vec - self.lr * mh / (np.sqrt(vh) + self.eps)


# ===========================================================================
#  Configurations de départ
# ===========================================================================

def grid_config(n, a, rng, jitter=1e-3):
    """Grille d'amorces adaptée au conteneur (carré : [0,a]², cercle : |x|<=a).

    Pour un conteneur carré, les objets sont répartis entre r et a-r sur chaque
    axe ; pour un conteneur circulaire, la grille est centrée en 0 et son pas
    est réduit si nécessaire pour tenir dans le disque. Ce n'est qu'une amorce :
    le polissage se charge des chevauchements éventuels."""
    c = _ctx()
    k = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / k))
    idx = np.arange(n)
    rr, cc_ = idx // k, idx % k
    if c['sqc']:
        if a >= k:
            ox, oy = (a - k) / 2.0, (a - rows) / 2.0
            xs = ox + np.arange(k) + 0.5
            ys = oy + np.arange(rows) + 0.5
        else:
            sp = max(0.0, a - 1.0) / max(1, k - 1)
            xs = 0.5 + np.arange(k) * sp
            ys = 0.5 + np.arange(rows) * sp
        centers = np.stack([xs[cc_], ys[rr]], axis=1)
    else:
        extent = c['r'] if c['cis'] else 0.5 * SQRT2
        span = (k - 1) * 0.5 * SQRT2
        s = (1.0 if span + extent <= a + 1e-12
             else max(0.0, a - extent) / max(1e-12, span))
        s = max(0.0, s)
        xs = (np.arange(k) - 0.5 * (k - 1)) * s
        ys = (np.arange(rows) - 0.5 * (rows - 1)) * s
        centers = np.stack([xs[cc_], ys[rr]], axis=1)
    centers = centers + rng.normal(0.0, jitter, centers.shape)
    thetas = np.zeros(n)
    return centers, thetas


def random_config(n, a, rng, rot=True):
    c = _ctx()
    if c['sqc']:
        lo, hi = c['r'], max(c['r'], a - c['r'])
        centers = rng.uniform(lo, hi, (n, 2))
    else:
        extent = c['r'] if c['cis'] else 0.5 * SQRT2
        rad = max(0.0, a - extent)
        ang = rng.uniform(0.0, 2.0 * math.pi, n)
        rr = rad * np.sqrt(rng.uniform(0.0, 1.0, n))
        centers = np.stack([rr * np.cos(ang), rr * np.sin(ang)], axis=1)
    thetas = rng.uniform(0.0, math.pi / 4.0, n) if rot else np.zeros(n)
    return centers, thetas


def ring_config(n, a, rng, rot=True, rings=3):
    """Amorce en anneaux concentriques (conteneur circulaire uniquement).

    Les empilements optimaux dans un disque placent souvent les objets en
    couronnes ; cette famille donne au polissage une topologie de départ
    crédible, que la grille et l'aléatoire n'explorent pas."""
    c = _ctx()
    extent = c['r'] if c['cis'] else 0.5 * SQRT2
    Rm = max(1e-6, a - extent)
    rings = max(1, min(int(rings), n))
    radii = Rm * (np.arange(1, rings + 1) / rings)
    caps = np.maximum(1, np.floor(2.0 * math.pi * radii)).astype(int)
    counts = np.zeros(rings, dtype=int)
    remaining = n
    for kk in range(rings - 1, -1, -1):
        if remaining <= 0:
            break
        counts[kk] = min(int(caps[kk]), remaining)
        remaining -= counts[kk]
    if remaining > 0:
        counts[-1] += remaining
    pos = []
    for kk in range(rings):
        m = counts[kk]
        if m == 0:
            continue
        ang0 = rng.uniform(0.0, 2.0 * math.pi)
        for t in range(m):
            ang = ang0 + 2.0 * math.pi * t / m
            pos.append((radii[kk] * math.cos(ang),
                        radii[kk] * math.sin(ang)))
    centers = np.array(pos[:n], dtype=float)
    centers += rng.normal(0.0, 0.01, centers.shape)
    thetas = rng.uniform(0.0, math.pi / 4.0, n) if rot else np.zeros(n)
    return centers, thetas


def hex_config(n, a, rng, rot=True):
    """Amorce en réseau triangulaire (hexagonal), puis mise à l'échelle pour
    tenir dans le conteneur.

    C'est la topologie dense naturelle pour des cercles (écartement 1 en
    horizontal, √3/2 en vertical : les voisins se touchent). On essaie toutes
    les formes de bloc (colonnes × lignes) et on garde celle qui demande le
    moins de compression."""
    c = _ctx()
    best = None
    for cols in range(1, n + 1):
        rows = int(math.ceil(n / cols))
        idx = np.arange(n)
        rr, cc_ = idx // cols, idx % cols
        x = cc_ + 0.5 * (rr % 2)
        y = rr * (math.sqrt(3.0) / 2.0)
        P = np.stack([x, y], axis=1)
        P -= 0.5 * (P.min(axis=0) + P.max(axis=0))
        half = 0.5 * (P.max(axis=0) - P.min(axis=0))
        if c['sqc']:
            room = max(1e-9, 0.5 * (a - 2.0 * c['r']))
            s = min(1.0, float(room / max(half.max(), 1e-9)))
        else:
            Rm = max(1e-9, a - c['r'])
            s = min(1.0, float(Rm / max(np.linalg.norm(P, axis=1).max(),
                                        1e-9)))
        if best is None or s > best[0]:
            best = (s, P, cols, rows)
    s, P, cols, rows = best
    P = P * s
    if c['sqc']:
        P += np.array([a / 2.0, a / 2.0])
    centers = P + rng.normal(0.0, 0.005, P.shape)
    thetas = np.zeros(n)
    return centers, thetas


def generic_families(n, a, rng, rot=True):
    """Familles d'amorces pour les modes sans conteneur carré + carrés.

    Les charpentes/palettes d'angles ne sont valides que dans [0,a]² : pour un
    disque (ou des cercles) on garde grilles, anneaux, réseau hexagonal et
    tirages aléatoires."""
    fam = [grid_config(n, a, rng, jitter=0.0),
           grid_config(n, a, rng, jitter=0.03),
           random_config(n, a, rng, rot),
           random_config(n, a, rng, rot)]
    if _CIRC_SHAPE:
        fam.append(hex_config(n, a, rng, rot))
        fam.append(hex_config(n, a, rng, rot))
    if _CIRC_CONT:
        for rings in (1, 2, 3, 4):
            fam.append(ring_config(n, a, rng, rot, rings))
    else:
        # grille décalée : lignes impaires translatées d'un demi-pas
        cc, th = grid_config(n, a, rng, jitter=0.01)
        k = max(1, int(math.ceil(math.sqrt(n))))
        rr = np.arange(n) // k
        cc[rr % 2 == 1, 0] += 0.5
        cc[:, 0] = np.clip(cc[:, 0], _R, a - _R)
        fam.append((cc, th))
    return fam


def _gobel_core(k):
    """Construction de Göbel EXACTE pour n = k²+1, côté k + √2/2.

    Base k=2 (n=5) : 4 coins + 1 carré à 45°.  Base k=3 (n=10) : cadre de 8
    carrés + rectangle 1×2 à 45°.  Récurrence k -> k+2 : on entoure la
    configuration de k-2 par un cadre d'épaisseur 1 (4 coins + k-2 carrés
    intérieurs par côté, soit 4k-4 carrés).  Le trou central a exactement le
    côté (k-2) + √2/2, donc l'intérieur reste valide ; les carrés du cadre
    sont alignés sur les axes et ne se chevauchent pas.

    Cette construction prouve C(k²+1) <= k + √2/2 pour tout k >= 2 (elle est
    optimale pour k=2 et k=3 ; pour k=4 elle est dominée par Bidwell)."""
    if k == 2:
        s = 2.0 + HALF_SQRT2
        centers = np.array([[0.5, 0.5], [s - 0.5, 0.5],
                            [0.5, s - 0.5], [s - 0.5, s - 0.5],
                            [s / 2.0, s / 2.0]], dtype=float)
        thetas = np.array([0.0, 0.0, 0.0, 0.0, math.pi / 4.0])
        return centers, thetas
    if k == 3:
        s = 3.0 + HALF_SQRT2
        centers = np.array([
            [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [s - 0.5, 0.5],
            [s - 0.5, s - 0.5], [s - 1.5, s - 0.5], [s - 0.5, s - 1.5],
            [0.5, s - 0.5],
            [s - 1.5, 1.5], [1.5, s - 1.5],
        ], dtype=float)
        thetas = np.zeros(10)
        thetas[8:] = math.pi / 4.0
        return centers, thetas

    inner_c, inner_t = _gobel_core(k - 2)
    inner_c = inner_c + 1.0                 # trou [1, k-1+√2/2]²
    L = (k - 2) + HALF_SQRT2                # côté intérieur
    S = k + HALF_SQRT2                      # côté extérieur
    m = k - 2                               # carrés par côté intérieur
    ring = []
    for x in (0.5, S - 0.5):                # 4 coins du cadre
        for y in (0.5, S - 0.5):
            ring.append((x, y))
    # m carrés centrés sur [1, 1+L] le long de chaque côté (les coins sont
    # déjà posés) ; m <= L donc les gaps restent positifs.
    pas = (L - 1.0) / (m - 1)
    pos = [1.5 + j * pas for j in range(m)]
    for x in pos:
        ring.append((x, 0.5))
        ring.append((x, S - 0.5))
    for y in pos:
        ring.append((0.5, y))
        ring.append((S - 0.5, y))
    centers = np.vstack([inner_c, np.array(ring, dtype=float)])
    thetas = np.concatenate([inner_t, np.zeros(len(ring))])
    return centers, thetas


def gobel_config(n, a, rng, rot=True, jitter=0.0):
    """Amorce EXACTE de Göbel généralisée pour n = k²+1 (k >= 2).

    À `a = k + √2/2` la configuration est sans chevauchement (contacts
    exacts) ; c'est un certificat direct de la borne d'acceptation.  Pour un
    `a` plus grand, elle est recentrée ; pour un `a` plus petit, elle est
    comprimée pour servir de topologie de départ au polissage.

    La construction utilise des carrés à 45° : elle n'est pas disponible si
    les rotations sont interdites (`rot=False`)."""
    k = math.isqrt(max(0, n - 1))
    if k < 2 or k * k + 1 != n or not rot:
        return None
    centers, thetas = _gobel_core(k)
    S = k + HALF_SQRT2
    if a >= S:
        centers = centers + 0.5 * (a - S)
    else:
        centers = centers * (a / S)
    if jitter:
        centers = centers + rng.normal(0.0, jitter, centers.shape)
        thetas = thetas + rng.normal(0.0, jitter, thetas.shape)
    return centers, thetas


def gobel_family(n, a, rng, rot=True):
    """Göbel exact + variante légèrement bruitée (si le cas s'applique)."""
    g = gobel_config(n, a, rng, rot, jitter=0.0)
    if g is None:
        return []
    fam = [g]
    g2 = gobel_config(n, a, rng, rot, jitter=1e-3)
    if g2 is not None:
        fam.append(g2)
    return fam


def trump_config(n, a, rng, rot=True, jitter=0.0):
    """Amorce EXACTE de Trump pour n = 11 (côté 3.877084, angles ≈ 40.18°).

    La configuration est construite à partir de la géométrie publiée
    (6 carrés alignés + 5 carrés pivotés). Avec `a >= côté optimal`, elle est
    simplement recentrée : la certification la resserre au côté optimal.
    Avec `a` plus petit, elle est comprimée pour servir de topologie de
    départ au polissage."""
    if n != 11 or not rot:
        return None
    s_opt = 3.87708359002281417730789706010096
    al = math.radians(40.1819372903297164652303423680606)
    ca, sa = math.cos(al), math.sin(al)
    x0 = 1.0 + 2.0 / ca - (s_opt - 2.0) * math.tan(al)
    r1 = 1.0 - (s_opt - 3.0) * ca
    u1 = ((1.0 + r1) * ca - 1.0) / sa
    v1 = ca - sa
    v2 = (s_opt - 1.0) / sa - r1 - (3.0 + u1) / math.tan(al)
    inner = [(0.5, 0.5, 0.0), (s_opt - 0.5, 0.5, 0.0),
             (x0 + 0.5, s_opt - 0.5, 0.0),
             (0.5, s_opt - 0.5, 0.0), (1.5, s_opt - 0.5, 0.0),
             (0.5, s_opt - 1.5, 0.0)]
    for (qx, qy) in ((0.5, 0.5), (u1 + 0.5, -0.5), (1.5, v1 + 0.5),
                     (u1 + 1.5, v1 - 0.5), (u1 + 2.5, -v2 + 0.5)):
        dx, dy = qx, qy - r1
        inner.append((ca * dx - sa * dy + 1.0,
                      sa * dx + ca * dy + 1.0, al))
    centers = np.array([[y, s_opt - x] for (x, y, _t) in inner], dtype=float)
    thetas = np.array([t for (_x, _y, t) in inner], dtype=float)
    if a >= s_opt:
        centers += 0.5 * (a - s_opt)
    else:
        centers *= a / s_opt
    if jitter:
        centers = centers + rng.normal(0.0, jitter, centers.shape)
        thetas = thetas + rng.normal(0.0, jitter, thetas.shape)
    return centers, thetas


def trump_family(n, a, rng, rot=True):
    g = trump_config(n, a, rng, rot, jitter=0.0)
    if g is None:
        return []
    fam = [g]
    g2 = trump_config(n, a, rng, rot, jitter=1e-3)
    if g2 is not None:
        fam.append(g2)
    return fam


def bidwell_config(n, a, rng, rot=True, jitter=0.0):
    """Amorce EXACTE de Bidwell pour n = 17 (côté 4.6755300936)."""
    if n != 17 or not rot:
        return None
    s_opt = 4.67553009360455095163411127048315
    a1 = math.radians(39.8049589797677950558662536965423)
    b1 = math.radians(36.6237863834465893201859433547298)
    x1 = 0.640176192429266087584168543270317
    v0 = 0.056838696870287317626412465614310
    u1 = 0.276426766820379851031162763090775
    u2 = 0.404478693276882795154318979238136
    x3 = 1.84732482651028509718122846547049
    y4 = 3.11346013251256404552110105703834
    r5 = 0.505927428006637436155510942971393
    inner = []
    # côté 1 : L de 3 carrés + carré imbriqué, puis son miroir
    for (x, y) in ((0.5, 1.5), (1.5, 0.5), (0.5, 0.5),
                   (0.5, s_opt - 0.5)):
        inner.append((x, y, 0.0))
    for (x, y) in ((s_opt - 0.5, 1.5), (s_opt - 1.5, 0.5),
                   (s_opt - 0.5, 0.5), (s_opt - 0.5, s_opt - 0.5)):
        inner.append((x, y, 0.0))
    # bloc de 6 carrés pivotés de a1
    ca, sa = math.cos(a1), math.sin(a1)
    qs = ((0.5, 0.5), (1.5, v0 + 0.5), (u1 + 0.5, -0.5),
          (u1 + 1.5, v0 - 0.5), (u2 + 0.5, -1.5),
          (u2 + 1.5, v0 - 1.5))
    for (qx, qy) in qs:
        inner.append((x1 + ca * qx - sa * qy,
                      2.0 + sa * qx + ca * qy, a1))
    # carré de bord + carré pivoté de -b1
    inner.append((x3 + 0.5, s_opt - 0.5, 0.0))
    cb, sb = math.cos(-b1), math.sin(-b1)
    dx, dy = 0.5, 0.5 - r5
    inner.append((x3 + 1.0 + cb * dx - sb * dy,
                  s_opt - 1.0 + sb * dx + cb * dy, -b1))
    # dernier carré de bord
    inner.append((s_opt - 0.5, y4 - 0.5, -math.pi / 2.0))
    # passage du repère du dessin (y vers le haut) au repère du solveur
    centers = np.array([(x, s_opt - y) for (x, y, _t) in inner],
                       dtype=float)
    # ATTENTION : ne PAS replier chaque angle dans [0, π/4] indépendamment.
    # Le repli t -> π/2 - t est une RÉFLEXION, pas une rotation : appliqué
    # seulement à certains carrés, il détruit la configuration exacte.
    thetas = np.array([-t for (_x, _y, t) in inner], dtype=float)
    if a >= s_opt:
        centers += 0.5 * (a - s_opt)
    else:
        centers *= a / s_opt
    if jitter:
        centers = centers + rng.normal(0.0, jitter, centers.shape)
        thetas = thetas + rng.normal(0.0, jitter, thetas.shape)
    return centers, thetas


def bidwell_family(n, a, rng, rot=True):
    g = bidwell_config(n, a, rng, rot, jitter=0.0)
    if g is None:
        return []
    fam = [g]
    g2 = bidwell_config(n, a, rng, rot, jitter=1e-3)
    if g2 is not None:
        fam.append(g2)
    return fam


def corners_config(n, a, rng, rot=True):
    """Amorce structurée : carrés aux 4 coins + restant au centre.

    Beaucoup d'empilements optimaux placent des carrés dans les coins (le cas
    n=5 de la Wikipédia : 4 coins + 1 carré pivoté à 45° au centre). La grille
    classique n'explore jamais cette topologie ; on lui donne ici.
    """
    anchors = [(0.5, 0.5), (a - 0.5, 0.5), (0.5, a - 0.5), (a - 0.5, a - 0.5)]
    k = min(4, n)
    pos = [anchors[i] for i in range(k)]
    rest = n - k
    if rest > 0:
        # petit bloc central (grille serrée) pour les carrés restants
        cols = max(1, int(math.ceil(math.sqrt(rest))))
        for i in range(rest):
            r, ccol = divmod(i, cols)
            pos.append((a / 2 + (ccol - (cols - 1) / 2) * 1.02,
                        a / 2 + (r - (cols - 1) / 2) * 1.02))
    centers = np.array(pos[:n], dtype=float)
    centers += rng.normal(0.0, 0.02, centers.shape)
    thetas = np.zeros(n)
    if rot:
        # le carré central est souvent pivoté (ex : n=5 -> 45°)
        for i in range(k, n):
            thetas[i] = math.pi / 4.0 if (i - k) % 2 == 0 else \
                rng.uniform(0.0, math.pi / 4.0)
    return centers, thetas


def stair_pair_config(n, a, rng, p, q, rot=True, jitter=0.01):
    """Amorce structurée « charpente » : 4 coins épinglés + escaliers + diagonale.

    - 4 carrés épinglés dans les coins ;
    - p carrés en escalier depuis le coin bas-gauche (le long du bord bas puis
      du bord gauche) ;
    - q carrés en escalier depuis le coin haut-droite (symétrique) ;
    - les carrés restants, par PAIRES miroirs (u, v) / (v, u) par rapport à la
      diagonale principale, pivotés de 45°.

    Cette charpente décrit les optimaux connus les plus « anguleux » :
    n = 5 (4 coins + 1 pivoté au centre, cas Göbel) et n = 10 (escaliers en L
    dans deux coins opposés + domino pivoté le long de la diagonale). Le balayage
    sur (p, q) (voir structured_family) évite d'incrustater un cas particulier :
    ce sont les gradients de polissage qui choisissent le membre pertinent.
    """
    if n < 4:
        return grid_config(n, a, rng, jitter=jitter)
    pos = [(0.5, 0.5), (a - 0.5, 0.5), (0.5, a - 0.5), (a - 0.5, a - 0.5)]
    p = max(0, min(p, n - 4))
    q = max(0, min(q, n - 4 - p))
    for i in range(p):
        if i % 2 == 0:
            pos.append((1.5 + i // 2, 0.5))
        else:
            pos.append((0.5, 1.5 + i // 2))
    for i in range(q):
        if i % 2 == 0:
            pos.append((a - 1.5 - i // 2, a - 0.5))
        else:
            pos.append((a - 0.5, a - 1.5 - i // 2))
    j = 0
    while len(pos) < n:
        u, v = 1.5 + j, a - 1.5 - j
        if n - len(pos) >= 2:
            pos.append((u, v))
            pos.append((v, u))
        else:
            pos.append((0.5 * (u + v), 0.5 * (u + v)))
        j += 1
    centers = np.array(pos[:n], dtype=float)
    centers += rng.normal(0.0, jitter, centers.shape)
    thetas = np.zeros(n)
    if rot:
        thetas[4 + p + q:] = math.pi / 4.0
    return centers, thetas


def structured_family(n, a, rng, rot=True, jitter=0.01, max_stair=4,
                      max_rest=4):
    """Balayage de la famille stair_pair_config : toutes les tailles
    d'escaliers (p, q) plausibles pour n carrés."""
    out = []
    for p in range(max_stair + 1):
        for q in range(max_stair + 1):
            rest = n - 4 - p - q
            if 4 + p + q > n or rest < 0 or rest > max_rest:
                continue
            out.append(stair_pair_config(n, a, rng, p, q, rot, jitter))
    return out


# ---------------------------------------------------------------------------
#  A2 — familles génératives supplémentaires (structures à motifs)
#  A6 — palette discrète d'angles (les angles des meilleurs empilements
#       connus sont 0°, 45°, atan(1/2), atan(1/3), ...) puis polissage
#       continu : on balaye d'abord le discret, l'Adam affine le réel.
# ---------------------------------------------------------------------------

ANGLE_PALETTE = (0.0, math.pi / 4.0, math.atan(0.5), math.atan(1.0 / 3.0),
                 math.pi / 6.0)


def diagonal_band_config(n, a, rng, rot=True, m=2, jitter=0.01):
    """Bande de m carrés pivotés à 45° le long de la diagonale principale,
    remplissage des deux coins opposés par des carrés alignés.

    Variante « bande » des charpentes stair_pair : plusieurs carrés pivotés
    en filet diagonal (construction de Göbel : n = k²+k se range comme ça)
    plutôt qu'un seul bloc central.
    """
    pos = []
    r = 0
    if m > 0:
        r = min(m, max(1, n))
        half = SQRT2 / 2.0
        if a - 2.0 * half < 0.5 or r == 1:
            band = [(0.5 * a, 0.5 * a)]
        else:
            band = [(half + (a - 2.0 * half) * t / (r - 1),) * 2
                    for t in range(r)]
        pos = list(band)
    # remplissage triangulaire des coins, en évitant la bande diagonale
    seen = set((round(x, 6), round(y, 6)) for (x, y) in pos)

    def _try(x, y):
        if x < 0.499 or y < 0.499 or x > a - 0.499 or y > a - 0.499:
            return False
        key = (round(x, 6), round(y, 6))
        if key in seen:
            return False
        for (bx, by) in pos:            # trop près de la bande pivotée ?
            if (x - bx) ** 2 + (y - by) ** 2 < 1.02 ** 2:
                return False
        seen.add(key)
        return True

    fill = []
    i = 0
    while len(fill) + len(pos) < n and i < 4 * int(a) + 16:
        for (x, y) in ((i + 0.5, 0.5), (0.5, i + 0.5),
                       (i + 0.5, i - 0.5), (i - 0.5, i + 0.5)):
            if len(fill) + len(pos) >= n:
                break
            if _try(x, y):
                fill.append((x, y))
        i += 1
    # miroir : cases symétriques par rapport à la diagonale (autre coin)
    mirror = [(a - x, a - y) for (x, y) in fill]
    seq = []
    for k in range(len(fill)):
        seq.append(fill[k])
        if len(seq) + len(pos) >= n:
            break
        if k < len(mirror) and _try(*mirror[k]):
            seq.append(mirror[k])
        if len(seq) + len(pos) >= n:
            break
    pos = pos + seq
    if len(pos) < n:                        # dernier recours : grille
        pos += [(0.5 + k % max(1, int(a)), 0.5 + k // max(1, int(a)))
                for k in range(n - len(pos))]
    centers = np.array(pos[:n], dtype=float)
    centers += rng.normal(0.0, jitter, centers.shape)
    thetas = np.zeros(n)
    if rot:
        for i in range(min(len(pos), r)):
            thetas[i] = math.pi / 4.0
    return centers, thetas


def frame_config(n, a, rng, rot=True, jitter=0.01):
    """Anneau périphérique (1 carré d'épaisseur le long des 4 bords) + bloc
    intérieur en grille.

    Topologie pertinente juste au-dessus d'un carré parfait (n ≈ a² - k) :
    les « déchets » vont sur le pourtour. Évaluée gratuitement comme les
    autres ; le polissage décide."""
    k = max(1, int(math.floor(a + 1e-9)))
    ring = []
    for i in range(k):                       # bord bas + haut
        ring.append((i + 0.5, 0.5))
        if len(ring) < n:
            ring.append((i + 0.5, a - 0.5))
    for j in range(1, k - 1):                # bord gauche + droite
        if len(ring) >= n:
            break
        ring.append((0.5, j + 0.5))
        if len(ring) < n:
            ring.append((a - 0.5, j + 0.5))
    pos = ring[:n]
    if len(pos) < n:                         # bloc intérieur pour le reste
        rest = n - len(pos)
        cols = max(1, int(math.ceil(math.sqrt(rest))))
        for i in range(rest):
            r_, c_ = divmod(i, cols)
            pos.append((a / 2 + (c_ - (cols - 1) / 2) * 1.02,
                        a / 2 + (r_ - (cols - 1) / 2) * 1.02))
    centers = np.array(pos[:n], dtype=float)
    centers += rng.normal(0.0, jitter, centers.shape)
    thetas = np.zeros(n)
    if rot:
        # angles aux angles de la bande (coin haut-droite) : topologie
        # « cadre + diagonale » observée sur plusieurs n
        for i in range(min(n, k)):
            if i % 2 == 1:
                thetas[i] = math.pi / 4.0
    return centers, thetas


def rows_rotblock_config(n, a, rng, rot=True, k=4, jitter=0.01):
    """Rangées alignées + un sous-bloc de k carrés pivotés à 45°.

    Les meilleurs empilements de n = 17..19 ressemblent à cela : une masse
    alignée + un « coin » pivoté qui libère la diagonale."""
    k = max(1, min(k, n))
    a_int = max(1, int(math.floor(a + 1e-9)))
    # le sous-bloc pivoté (en filet le long du coin haut-droite)
    half = SQRT2 / 2.0
    block = []
    for t in range(k):
        block.append((min(a - half, a - half - 0.7 * (t % 3)),
                      min(a - half, a - half - 0.7 * (t // 3))))
    rest = n - k
    cols = max(1, min(a_int, rest if rest else 1))
    pos = []
    for i in range(rest):
        r_, c_ = divmod(i, cols)
        pos.append((0.5 + c_ if 0.5 + c_ <= a - 0.5 else a - 0.5,
                    0.5 + r_ if 0.5 + r_ <= a - 0.5 else a - 0.5))
    pos = pos + block
    centers = np.array(pos[:n], dtype=float)
    centers += rng.normal(0.0, jitter, centers.shape)
    thetas = np.zeros(n)
    if rot:
        thetas[rest:] = math.pi / 4.0
    return centers, thetas


def palette_config(n, a, rng, rot=True, theta=0.0, pattern=0, jitter=0.01):
    """Grille de positions avec angles imposés par la palette discrète (A6).

    pattern 0 : tous à `theta` ; 1 : damier theta/0 ; 2 : lignes alternées ;
    3 : bloc supérieur à theta, inférieur à 0. Le polissage continu affine
    ensuite les angles au contact exact."""
    c0, t0 = grid_config(n, a, rng, jitter=0.0)
    if not rot:
        return c0 + rng.normal(0.0, jitter, c0.shape), t0
    cols = max(1, int(math.floor(a + 1e-9)))
    idx = np.arange(n)
    r_, c_ = idx // cols, idx % cols
    th = np.zeros(n)
    if pattern == 0:
        th[:] = theta
    elif pattern == 1:
        th = np.where((r_ + c_) % 2 == 0, theta, 0.0)
    elif pattern == 2:
        th = np.where(r_ % 2 == 0, theta, 0.0)
    else:
        th = np.where(r_ >= r_.max() / 2.0, theta, 0.0)
    return c0 + rng.normal(0.0, jitter, c0.shape), th


def sweep_families(n, a, rng, rot=True, jitter=0.01):
    """A2/A6 — balayage complet des familles structurées, évaluation EXACTE
    gratuite de chaque membre (la tri décide qui sera poli) :
    charpentes (p, q) + bandes diagonales + cadres + rangées/bloc pivoté +
    palette discrète d'angles. ~40 configurations, ~ms."""
    fam = structured_family(n, a, rng, rot, jitter)
    if rot:
        for m in (1, 2, 3, 4):
            if m <= n:
                fam.append(diagonal_band_config(n, a, rng, True, m, jitter))
        fam.append(frame_config(n, a, rng, True, jitter))
        fam.append(frame_config(n, a, rng, False, jitter))
        for k in (max(1, n // 5), max(1, n // 3)):
            fam.append(rows_rotblock_config(n, a, rng, True, k, jitter))
            fam.append(rows_rotblock_config(n, a, rng, False, k, jitter))
        for theta in ANGLE_PALETTE:
            for pattern in (0, 1):
                fam.append(palette_config(n, a, rng, True, theta, pattern,
                                          jitter))
        fam.append(palette_config(n, a, rng, True, ANGLE_PALETTE[1], 2,
                                  jitter))
    else:
        fam.append(diagonal_band_config(n, a, rng, False, 2, jitter))
        fam.append(frame_config(n, a, rng, False, jitter))
        fam.append(rows_rotblock_config(n, a, rng, False, max(1, n // 4),
                                        jitter))
    return fam


def fold_theta(th):
    """Ramène theta dans [0, pi/4] (orientations uniques d'un carré)."""
    t = np.mod(th, math.pi / 2.0)
    return np.where(t > math.pi / 4.0, math.pi / 2.0 - t, t)


def anneal_config(n, a, rng, iters=1500, rot=True,
                  T0=0.05, T1=2e-4, tau=1e-3, start=None):
    """Trempe simulée (Monte-Carlo sur x, y, theta, perte lisse).

    Amorce exploratoire complémentaire du réseau et du polissage : saute
    entre bassins d'attraction que le gradient lisse ne rejoint jamais.
    """
    if start is None:
        c, th = random_config(n, a, rng, rot)
    else:
        c, th = start[0].copy(), start[1].copy()
    th = fold_theta(th)
    L, _, _, _ = loss_and_backward(c, th, a, tau)
    best = (exact_violation(c, th, a), c.copy(), th.copy())
    step = 0.15 * a / math.sqrt(n)
    acc = 0
    for t in range(iters):
        T = T0 * (T1 / T0) ** (t / max(1, iters - 1))
        cn, thn = c.copy(), th.copy()
        k = int(rng.integers(1, 4))
        pick = rng.choice(n, size=min(k,n), replace=False)
        cn[pick] += rng.normal(0.0, step, (len(pick), 2))
        if rot:
            thn[pick] += rng.normal(0.0, step * 0.5, len(pick))
            thn = fold_theta(thn)
        Ln, _, _, _ = loss_and_backward(cn, thn, a, tau)
        dE = Ln - L
        if dE <= 0.0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
            c, th, L = cn, thn, Ln
            acc += 1
        v = exact_violation(c, th, a)
        if v < best[0]:
            best = (v, c.copy(), th.copy())
        if (t + 1) % 100 == 0:
            rate = acc / 100.0
            step *= 1.25 if rate > 0.45 else (0.8 if rate < 0.2 else 1.0)
            step = float(np.clip(step, 1e-3, 0.5 * a))
            acc = 0
    return best


# ===========================================================================
#  Entraînement de l'agent (Adam sur la perte lisse, ou Evolution Strategies)
# ===========================================================================

def train_agent_adam(params, n, a, base, rot, iters, rng,
                     tau0=0.05, tau_min=1e-4, lr=1e-2, lam_p=1.0, lam_v=1.0):
    adam = Adam(lr=lr)
    best = (float("inf"), None, None)
    for t in range(iters):
        tau = max(tau_min, tau0 * (tau_min / tau0) ** (t / max(1, iters - 1)))
        X = build_inputs(params, n, a, base)
        centers, thetas, cache, _ = agent_config(params, X, base, rot)
        L, dc, dt, _ = loss_and_backward(centers, thetas, a, tau, lam_p, lam_v)
        O_grad = np.concatenate([dc, dt[:, None]], axis=1)
        grads, dX = set_backward(params, cache, O_grad)
        dE = np.zeros_like(params["E"])
        dE[:n] = dX[:, :params["E"].shape[1]]   # gradient vers les embeddings
        grads["E"] = dE
        vec = ravel_params(params)
        gvec = np.concatenate([grads[k].ravel() for k in sorted(grads)])
        newvec = adam.step(vec, gvec)
        params_new = unravel_params(newvec, params)
        params.clear()
        params.update(params_new)
        v = exact_violation(centers, thetas, a)
        if v < best[0]:
            best = (v, centers.copy(), thetas.copy())
    return best


def train_agent_es(params, n, a, base, rot, iters, rng,
                   pop=16, sigma=0.05, lr=3e-2, lam_p=1.0, lam_v=1.0):
    """Evolution Strategies (OpenAI-ES) : recherche de politique sans gradient."""
    tau_min = 1e-4

    def score(vec):
        p = unravel_params(vec, params)
        Xp = build_inputs(p, n, a, base)
        c, th, _, _ = agent_config(p, Xp, base, rot)
        L, _, _, _ = loss_and_backward(c, th, a, tau_min, lam_p, lam_v)
        return -L                                     # récompense

    vec = ravel_params(params)
    adam = Adam(lr=lr)
    best = (float("inf"), None, None)
    for _ in range(iters):
        eps = rng.normal(0.0, 1.0, (pop // 2, vec.size))
        R = np.empty(pop)
        for k in range(pop // 2):
            R[2 * k] = score(vec + sigma * eps[k])
            R[2 * k + 1] = score(vec - sigma * eps[k])
        R = (R - R.mean()) / (R.std() + 1e-8)
        grad = (R[:, None] * np.concatenate([eps, -eps])).sum(axis=0) \
            / (pop * sigma)
        vec = adam.step(vec, -grad)
        p = unravel_params(vec, params)
        Xp = build_inputs(p, n, a, base)
        c, th, _, _ = agent_config(p, Xp, base, rot)
        v = exact_violation(c, th, a)
        if v < best[0]:
            best = (v, c.copy(), th.copy())
    final = unravel_params(vec, params)   # AVANT d'effacer (gabarit de formes)
    params.clear()
    params.update(final)
    return best


# ===========================================================================
#  Polissage : raffinement direct des coordonnées (x, y, theta)
# ===========================================================================

def rescale_to_feasible(centers, thetas, a):
    """Convertit une config quasi valide en empilement PROUVÉ à une taille a'.

    On dilate homothétiquement la configuration (centres × (1+δ)) et le
    conteneur (a × (1+δ)), en gardant la taille des objets. Les distances
    entre centres croissent, donc les chevauchements disparaissent ; la
    contenance reste vraie dans les 4 modes :
    - carré : les points extrêmes restent dans [0, a']² ;
    - cercle : |point| <= a' (ou |centre| + r <= a').
    On cherche le plus petit δ par bissection -> empilement exactement valide
    à a' = a(1+δ)."""

    v = exact_violation(centers, thetas, a)
    if v <= 0.0:
        return a, centers, thetas
    lo, hi = 0.0, max(1e-6, 8.0 * v)
    while exact_violation(centers * (1 + hi), thetas, a * (1 + hi)) > 0.0 \
            and hi < 1.0:
        hi *= 2.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if exact_violation(centers * (1 + mid), thetas, a * (1 + mid)) <= 0.0:
            hi = mid
        else:
            lo = mid
    delta = hi * (1 + 1e-12)
    return a * (1 + delta), centers * (1 + delta), thetas


# ---------------------------------------------------------------------------
#  A1 — contacts exacts : sertissement du conteneur + snapping des contacts
# ---------------------------------------------------------------------------

def vertex_coords(centers, thetas):
    """Points extrêmes (n, 4, 2) : sommets d'un carré, points cardinaux d'un
    cercle (centre ± r sur chaque axe)."""
    return build_geometry(centers, thetas, 0.0)["V"]


def tighten_a(centers, thetas, a):
    """Plus petit conteneur EXACT pour CETTE configuration (sertissement).

    Les contraintes de chevauchement ne dépendent pas de a. On translate la
    configuration pour la centrer sur ses points extrêmes, puis :
    - conteneur carré : côté = max(largeur, hauteur) ;
    - conteneur cercle : rayon = max distance des points extrêmes (ou des
      centres + r pour des cercles) au centre de la boîte englobante.
    Le résultat n'est accepté que s'il est réellement faisable (vérification
    exacte) : c'est donc toujours un certificat rigoureux."""
    if exact_violation(centers, thetas, a) > FEAS_TOL:
        return a, centers, thetas
    c = _ctx()
    P = vertex_coords(centers, thetas)
    flat = P.reshape(-1, 2)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    if c['sqc']:
        M = float(max(hi - lo))
        c2 = centers - lo
        if 1e-12 < M < a and exact_violation(c2, thetas, M) <= FEAS_TOL:
            return M, c2, thetas
        return a, centers, thetas
    mid = 0.5 * (lo + hi)
    c2 = centers - mid
    if c['cis']:
        R = float(np.linalg.norm(c2, axis=1).max()) + c['r']
    else:
        R = float(np.linalg.norm((P - mid).reshape(-1, 2), axis=1).max())
    if 1e-12 < R < a and exact_violation(c2, thetas, R) <= FEAS_TOL:
        return R, c2, thetas
    return a, centers, thetas


def snap_config(centers, thetas, a, rot=True, tol=8e-3, max_eq=800):
    """Ajustement des contacts actifs par moindres carrés (Levenberg-Marquardt).

    Équations actives, selon le mode :
    - gap = 0 entre paires quasi en contact (axe SAT pour des carrés,
      distance - 2r pour des cercles) ;
    - conteneur carré : point extrême = 0 ou = a ;
    - conteneur circulaire : |point extrême| = a (sommets de carrés) ou
      |centre| + r = a (cercles).
    La jacobienne est creuse : chaque équation ne dépend que des variables de
    « son » objet (et de son voisin). Les contraintes inactives restent
    strictement satisfaites.

    Renvoie (c, θ) affinés si la violation exacte s'améliore, sinon None.
    """
    try:
        from scipy.optimize import least_squares
        from scipy.sparse import csr_matrix
    except ImportError:
        return None
    c = _ctx()
    if c['cis'] or not c['sqs']:
        rot = False            # l'orientation d'un cercle est sans effet
    n = centers.shape[0]
    v0 = exact_violation(centers, thetas, a)
    nv = 2 * n + (n if rot else 0)

    def unpack(x):
        c2 = x[:2 * n].reshape(n, 2)
        th2 = x[2 * n:] if rot else thetas
        return c2, th2

    c_cur, th_cur, v_cur = centers, thetas, v0

    for _round in range(5):
        geo = build_geometry(c_cur, th_cur, a)
        V = geo["V"]
        eqs = []          # ('g',i,j,k) ('b',i,q,cd,t) ('r',i,q,0,0) ('rc',i,0,0,0)
        iu = np.triu_indices(n, k=1)
        if len(iu[0]):
            gmax = geo["g"][iu].max(axis=1)
            for idx in range(len(gmax)):
                if gmax[idx] <= tol:
                    i, j = int(iu[0][idx]), int(iu[1][idx])
                    eqs.append(("g", i, j, int(np.argmax(geo["g"][i, j]))))
        if c['sqc']:
            for i in range(n):
                for q in range(4):
                    for cd in (0, 1):
                        val = V[i, q, cd]
                        if val <= tol:
                            eqs.append(("b", i, q, cd, 0.0))
                        elif val >= a - tol:
                            eqs.append(("b", i, q, cd, a))
        elif not c['cis']:
            for i in range(n):
                for q in range(4):
                    rv = math.hypot(V[i, q, 0], V[i, q, 1])
                    if abs(rv - a) <= tol:
                        eqs.append(("r", i, q, 0, 0.0))
        else:
            for i in range(n):
                rv = math.hypot(c_cur[i, 0], c_cur[i, 1]) + c['r']
                if abs(rv - a) <= tol:
                    eqs.append(("rc", i, 0, 0, 0.0))
        if not eqs:
            break

        def activity(e):
            if e[0] == "g":
                return abs(geo["g"][e[1], e[2], e[3]])
            if e[0] == "b":
                return abs(V[e[1], e[2], e[3]] - e[4])
            if e[0] == "r":
                return abs(math.hypot(V[e[1], e[2], 0],
                                      V[e[1], e[2], 1]) - a)
            return abs(math.hypot(c_cur[e[1], 0], c_cur[e[1], 1])
                       + c['r'] - a)

        if len(eqs) > max_eq:
            eqs = sorted(eqs, key=activity)[:max_eq]

        gi = np.array([e[1] for e in eqs if e[0] == "g"], dtype=np.int64)
        gj = np.array([e[2] for e in eqs if e[0] == "g"], dtype=np.int64)
        gk = np.array([e[3] for e in eqs if e[0] == "g"], dtype=np.int64)
        bi = np.array([e[1] for e in eqs if e[0] == "b"], dtype=np.int64)
        bq = np.array([e[2] for e in eqs if e[0] == "b"], dtype=np.int64)
        bd = np.array([e[3] for e in eqs if e[0] == "b"], dtype=np.int64)
        bt = np.array([e[4] for e in eqs if e[0] == "b"], dtype=float)
        ri = np.array([e[1] for e in eqs if e[0] == "r"], dtype=np.int64)
        rq = np.array([e[2] for e in eqs if e[0] == "r"], dtype=np.int64)
        rci = np.array([e[1] for e in eqs if e[0] == "rc"], dtype=np.int64)
        ng, nb, nr, nrc = gi.size, bi.size, ri.size, rci.size

        def resid(x):
            c2, th2 = unpack(x)
            geo2 = build_geometry(c2, th2, a)
            out = np.empty(len(eqs))
            pos = 0
            if ng:
                out[:ng] = geo2["g"][gi, gj, gk]
                pos = ng
            if nb:
                out[pos:pos + nb] = geo2["V"][bi, bq, bd] - bt
                pos += nb
            if nr:
                V2 = geo2["V"]
                out[pos:pos + nr] = np.hypot(V2[ri, rq, 0],
                                             V2[ri, rq, 1]) - a
                pos += nr
            if nrc:
                out[pos:] = np.hypot(c2[rci, 0], c2[rci, 1]) + c['r'] - a
            return out

        def jac(x):
            """Jacobienne ANALYTIQUE creuse des équations de contact.

            Chaque équation ne dépend que des variables de son objet (et de
            son voisin pour un contact) : on assemble directement les valeurs
            non nulles, sans différences finies internes à least_squares."""
            c2, th2 = unpack(x)
            thx = th2 if rot else thetas
            rows, cols, vals = [], [], []

            def put(e, idx, val):
                if val != 0.0:
                    rows.append(e)
                    cols.append(idx)
                    vals.append(val)

            # Au voisinage EXACT de pi/2, |cos| a une pointe : la dérivée
            # d'un côté vaut +1/2, de l'autre -1/2.  Pour la fonction
            # composée (support(tj - phi) avec phi = ti + pi/2), la dérivée
            # totale est nulle.  Un signe numérique quasi nul donnerait
            # artificiellement 1/2 : on choisit le sous-gradient 0 aux pointes.
            def _sgrad(al):
                ca, sa = math.cos(al), math.sin(al)
                sga = 0.0 if abs(ca) < 1e-12 else math.copysign(1.0, ca)
                sgb = 0.0 if abs(sa) < 1e-12 else math.copysign(1.0, sa)
                return 0.5 * (-sga * sa + sgb * ca)

            for e, eq in enumerate(eqs):
                kind = eq[0]
                if kind == "g":
                    _, i, j, k = eq
                    if c['cis']:
                        # cercle : g = |cj - ci| - 2r
                        dx = c2[j, 0] - c2[i, 0]
                        dy = c2[j, 1] - c2[i, 1]
                        dist = max(math.hypot(dx, dy), 1e-15)
                        nx, ny = dx / dist, dy / dist
                        put(e, 2 * i, -nx)
                        put(e, 2 * i + 1, -ny)
                        put(e, 2 * j, nx)
                        put(e, 2 * j + 1, ny)
                        continue
                    ti, tj = thx[i], thx[j]
                    phi = (ti if k < 2 else tj) + (HALF_PI if k % 2 else 0.0)
                    ux, uy = math.cos(phi), math.sin(phi)
                    dx, dy = c2[j, 0] - c2[i, 0], c2[j, 1] - c2[i, 1]
                    p = dx * ux + dy * uy
                    s = 0.0 if p == 0.0 else (1.0 if p > 0.0 else -1.0)
                    # u_perp = (-sin phi, cos phi)
                    dphi = -dx * uy + dy * ux
                    hpi = _sgrad(ti - phi)
                    hpj = _sgrad(tj - phi)
                    put(e, 2 * i, -s * ux)
                    put(e, 2 * i + 1, -s * uy)
                    put(e, 2 * j, s * ux)
                    put(e, 2 * j + 1, s * uy)
                    if rot:
                        if k < 2:
                            put(e, 2 * n + i, s * dphi + hpj)
                            put(e, 2 * n + j, -hpj)
                        else:
                            put(e, 2 * n + i, -hpi)
                            put(e, 2 * n + j, s * dphi + hpi)
                elif kind == "b":
                    _, i, q, cd, _t = eq
                    ox, oy = OFF[q]
                    th_i = thx[i]
                    ct, st = math.cos(th_i), math.sin(th_i)
                    put(e, 2 * i + cd, 1.0)
                    if rot:
                        if cd == 0:
                            put(e, 2 * n + i, -ox * st - oy * ct)
                        else:
                            put(e, 2 * n + i, ox * ct - oy * st)
                elif kind == "r":
                    _, i, q, _z, _t = eq
                    ox, oy = OFF[q]
                    th_i = thx[i]
                    ct, st = math.cos(th_i), math.sin(th_i)
                    vx = c2[i, 0] + ox * ct - oy * st
                    vy = c2[i, 1] + ox * st + oy * ct
                    nv_v = max(math.hypot(vx, vy), 1e-15)
                    nx, ny = vx / nv_v, vy / nv_v
                    put(e, 2 * i, nx)
                    put(e, 2 * i + 1, ny)
                    if rot:
                        put(e, 2 * n + i,
                            nx * (-ox * st - oy * ct)
                            + ny * (ox * ct - oy * st))
                else:  # "rc" : |centre| + r - a
                    _, i, _q, _z, _t = eq
                    cx, cy = c2[i, 0], c2[i, 1]
                    nc = max(math.hypot(cx, cy), 1e-15)
                    put(e, 2 * i, cx / nc)
                    put(e, 2 * i + 1, cy / nc)
            return csr_matrix((vals, (rows, cols)), shape=(len(eqs), nv))

        x0 = np.concatenate([c_cur.ravel(),
                             th_cur.copy() if rot else np.zeros(0)])
        try:
            sol = least_squares(resid, x0, jac=jac, method="trf",
                                xtol=1e-11, ftol=1e-11, gtol=1e-11,
                                max_nfev=120)
        except Exception:
            break
        c2, th2 = unpack(sol.x)
        v2 = exact_violation(c2, th2, a)
        if v2 <= FEAS_TOL:
            # Même quand la config était déjà valide, on accepte le résultat
            # s'il permet un conteneur STRICTEMENT plus petit (contacts exacts).
            a_cur_cert, _, _ = tighten_a(c_cur, th_cur, a)
            a_new_cert, _, _ = tighten_a(c2, th2, a)
            if v0 > FEAS_TOL or a_new_cert < a_cur_cert - 1e-12:
                return c2, th2
            break
        if v2 < v_cur:
            c_cur, th_cur, v_cur = c2, th2, v2
        else:
            break

    if v_cur < v0:
        return c_cur, th_cur
    return None


def certify(centers, thetas, a, rot=True, snap=True, snap_tol=2e-2):
    """Certification complète d'une configuration : snapping (si léger
    chevauchement) → dilatation rigoureuse (rescale_to_feasible) →
    sertissement exact du conteneur (tighten_a).

    Renvoie (a_certifié, c, th) : empilement PROUVÉ valide à a_certifié.
    """
    v = exact_violation(centers, thetas, a)
    if snap and v <= snap_tol:
        snapped = snap_config(centers, thetas, a, rot=rot)
        if snapped is not None:
            centers, thetas = snapped
    a1, c1, t1 = rescale_to_feasible(centers, thetas, a)
    return tighten_a(c1, t1, a1)


def peel_refine(centers, thetas, a, rng, rot=True, rounds=4,
                iters=150, hard_iters=80, deadline=None):
    """Raffinement final « à pelures » : resserre le conteneur par-dessous la
    bissection (qui s'arrête à ±tol) en partant de la config certifiée.

    Boucle : certifie → tente a·(1-δ) avec δ décroissant → re-polit la config
    chaude dans ce conteneur plus serré → recertifie. Tant que le côté
    certifié baisse, on recommence. Chaque tour est un petit essai local,
    très peu coûteux comparé à un essai complet.
    """
    best_a, bc, bt = certify(centers, thetas, a, rot)
    n = bc.shape[0]
    for _ in range(rounds):
        if deadline is not None and time.time() >= deadline:
            break
        progressed = False
        for frac in (0.01, 0.004, 0.0015, 5e-4, 2e-4, 8e-5, 3e-5, 1e-5,
                     4e-6, 1.5e-6, 5e-7):
            if deadline is not None and time.time() >= deadline:
                break
            a_try = best_a * (1.0 - frac)
            if a_try <= area_lower_bound(n) + 1e-9:   # borne d'aire : inutile
                continue
            _v, c2, t2 = polish((bc, bt), a_try, iters, rng, rot,
                                hard_iters=hard_iters)
            a3, c3, t3 = certify(c2, t2, a_try, rot)
            if a3 < best_a - 1e-12:
                best_a, bc, bt = a3, c3, t3
                progressed = True
                break
            # Compression directe : le snapping peut résoudre les contacts
            # actifs dans le conteneur rétréci même quand Adam n'y arrive pas.
            snapped = snap_config(bc, bt, a_try, rot=rot)
            if snapped is not None:
                a4, c4, t4 = certify(snapped[0], snapped[1], a_try, rot)
                if a4 < best_a - 1e-12:
                    best_a, bc, bt = a4, c4, t4
                    progressed = True
                    break
        if not progressed:
            break
    return best_a, bc, bt


def scipy_refine(centers, thetas, a, rot=True, iters=40, tau=1e-6,
                 lam_p=1.0, lam_v=1.0):
    """B5 — quasi-newton L-BFGS-B (scipy) avec gradient ANALYTIQUE exact.

    Adam progresse lentement en fin de course (moments, pas adaptatif) ;
    L-BFGS-B reconstruit une approximation de la hessienne : quelques dizaines
    d'itérations suffisent souvent à convertir un optimum « presque » faisable
    (v ~ 1e-4..1e-2) en config exactement valide — ce qui évite la dilatation
    coûteuse lors de la certification. Renvoie (v, c, th) ou None (scipy
    absent / échec), pour ne JAMAIS aggraver le résultat courant.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    n = centers.shape[0]
    x0 = np.concatenate([centers.ravel(),
                         thetas.copy() if rot else np.zeros(0)])

    def fg(x):
        c = x[:2 * n].reshape(n, 2)
        th = x[2 * n:] if rot else thetas
        L, dc, dt, _ = loss_and_backward(c, th, a, tau, lam_p, lam_v)
        g = np.concatenate([dc.ravel(), dt if rot else dt * 0.0])
        return L, g

    try:
        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": iters, "maxls": 24})
    except Exception:
        return None
    x = np.asarray(res.x, dtype=float)
    c2 = x[:2 * n].reshape(n, 2)
    th2 = x[2 * n:] if rot else thetas
    return exact_violation(c2, th2, a), c2, th2


def polish(cfg, a, iters, rng, rot=True, lr=3e-3, tau0=1e-3, tau_min=1e-5,
           lam_p=1.0, lam_v=1.0, hard_iters=120):
    centers, thetas = cfg[0].copy(), cfg[1].copy()
    if not rot:
        thetas[:] = 0.0
    n = centers.shape[0]
    var = np.concatenate([centers.ravel(), thetas])
    adam = Adam(lr=lr)
    best = (float("inf"), centers.copy(), thetas.copy())
    for t in range(iters):
        tau = max(tau_min, tau0 * (tau_min / tau0) ** (t / max(1, iters - 1)))
        c = var[:2 * n].reshape(-1, 2)
        th = var[2 * n:]
        L, dc, dt, _ = loss_and_backward(c, th, a, tau, lam_p, lam_v)
        gvar = np.concatenate([dc.ravel(), dt if rot else dt * 0.0])
        var = adam.step(var, gvar)
        c = var[:2 * n].reshape(-1, 2)
        th = var[2 * n:]
        v = exact_violation(c, th, a)
        if v < best[0]:
            best = (v, c.copy(), th.copy())
    # --- phase « dure » : tau -> 0, on obtient le sous-gradient exact de la
    #     violation (axe argmax), ce qui permet d'atteindre le contact exact.
    var = np.concatenate([best[1].ravel(), best[2]])
    adam = Adam(lr=lr * 0.5)
    for t in range(hard_iters):
        tau = 1e-9
        c = var[:2 * n].reshape(-1, 2)
        th = var[2 * n:]
        L, dc, dt, _ = loss_and_backward(c, th, a, tau, lam_p, lam_v)
        gvar = np.concatenate([dc.ravel(), dt if rot else dt * 0.0])
        var = adam.step(var, gvar)
        c = var[:2 * n].reshape(-1, 2)
        th = var[2 * n:]
        v = exact_violation(c, th, a)
        if v < best[0]:
            best = (v, c.copy(), th.copy())
    # --- B5 : dernier coup de quasi-newton (L-BFGS-B) si pas encore faisable
    if best[0] > 0.0:
        r = scipy_refine(best[1], best[2], a, rot=rot, tau=tau_min,
                         lam_p=lam_p, lam_v=lam_v)
        if r is not None and r[0] < best[0]:
            best = r
    return best


# ===========================================================================
#  Un essai : agents (réseau) + candidats issus des champions + polissage
# ===========================================================================

def _train_worker(task):
    """Tâche parallèle : entraîner UN agent sur sa propre amorce."""
    (bc, bth, seed, n, a, iters, rot, opt, max_slots, de, H, arch) = task
    rr = np.random.default_rng(int(seed))
    params = init_agent(rr, max(int(max_slots), n), de, H, arch)
    if opt == "es":
        v, c, th = train_agent_es(params, n, a, (bc, bth), rot, iters, rr)
    else:
        v, c, th = train_agent_adam(params, n, a, (bc, bth), rot, iters, rr)
    return (v, c, th, bc, bth)


def _polish_worker(task):
    """Tâche parallèle : polir UNE configuration (indépendante des autres)."""
    c, th, a, seed, iters, rot = task
    return polish((c, th), a, iters, np.random.default_rng(int(seed)), rot)


def _anneal_worker(task):
    """Tâche parallèle : UNE trempe simulée (départ aléatoire ou en grille)."""
    n, a, seed, iters, rot, kind = task
    rr = np.random.default_rng(int(seed))
    start = grid_config(n, a, rr, jitter=0.02) if kind == 1 else None
    return anneal_config(n, a, rr, iters=iters, rot=rot, start=start)


# ---------------------------------------------------------------------------
#  A5 — sélection DIVERSE des candidats à polir (histogramme de distances
#       inter-paires) : évite de polir dix fois la même topologie.
# ---------------------------------------------------------------------------

def descriptor(centers, thetas, a):
    """Descripteur de topologie (A5) : histogramme des distances inter-paires
    (16 boîtes) + quelques stats d'angles normalisées."""
    n = centers.shape[0]
    if n < 2:
        return np.zeros(19)
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    h, _ = np.histogram(d[iu], bins=16, range=(0.0, a * 1.4143))
    h = h.astype(float)
    s = h.sum()
    if s > 0:
        h /= s
    ang = np.array([thetas.mean() * 4.0 / math.pi,
                    thetas.max() * 4.0 / math.pi,
                    float((np.abs(fold_theta(thetas)) < 1e-9).mean())])
    return np.concatenate([0.5 * h, 0.2 * ang])


def diverse_top(scored, a, k, thresh=0.12):
    """A5 — sélection des k meilleurs candidats à polir, en écartant les
    quasi-doublons topologiques (distance L1 de descripteurs < thresh).
    Le meilleur est toujours retenu ; si la diversité ne remplit pas k,
    on complète dans l'ordre du score."""
    if len(scored) <= k:
        return list(scored)
    descs = [descriptor(c, th, a) for (_, c, th) in scored]
    chosen = [0]
    for i in range(1, len(scored)):
        if len(chosen) >= k:
            break
        if min(np.abs(descs[i] - descs[j]).sum() for j in chosen) >= thresh:
            chosen.append(i)
    i = 1
    while len(chosen) < k and i < len(scored):
        if i not in chosen:
            chosen.append(i)
        i += 1
    return [scored[i] for i in sorted(chosen)]


def attempt(n, a, rng, agents=6, iters=250, polish_iters=300,
            champions=(), rot=True, opt="adam", max_slots=64, de=16, H=64,
            quiet=True, anneal_iters=1500, kicks=3, arch="set", deadline=None):
    """Tente de ranger n objets dans le conteneur de taille `a` (4 modes).

    Retourne (viol_exacte, centers, thetas).  viol <= FEAS_TOL  <=>  faisable.
    Les agents et les polissages sont répartis sur le pool de processus
    (setup_pool) quand'il existe. Archi : `set` (Deep-Set équivariant, défaut
    A4).  `deadline` (time.time) interrompt les phases optionnelles : on
    renvoie alors le meilleur candidat déjà disponible.
    """
    a = float(a)
    c = _ctx()
    candidates = []
    next_seed = lambda: int(rng.integers(0, 2 ** 62))

    def expired():
        return deadline is not None and time.time() >= deadline

    if expired():
        c0, th0 = grid_config(n, a, rng, jitter=0.0)
        return exact_violation(c0, th0, a), c0, th0

    # --- candidats « champions » (configurations déjà connues), rééchelonnées
    for (cc, cth, ca) in champions:
        if ca <= 0:
            continue
        scale = a / ca
        c2 = cc * scale
        if c['sqc']:
            c2 = np.clip(c2, c['r'], max(c['r'], a - c['r']))
        else:
            extent = c['r'] if c['cis'] else 0.5 * SQRT2
            lim = max(1e-9, a - extent)
            norms = np.linalg.norm(c2, axis=1)
            fac = np.minimum(1.0, lim / np.maximum(norms, 1e-12))
            c2 = c2 * fac[:, None]
        c2 = c2 + rng.normal(0, 0.01, cc.shape)
        th2 = cth + (rng.normal(0, 0.01, cth.shape) if rot else 0.0)
        candidates.append((c2, th2))

    # --- l'IA : population d'agents (réseau) entraînés, amorces variées.
    #     Les agents sont INDÉPENDANTS -> entraînés en parallèle (pool).
    if c['sqc'] and not c['cis']:
        inits = [
            lambda nn, aa, rr, r: grid_config(nn, aa, rr, jitter=0.02),
            lambda nn, aa, rr, r: corners_config(nn, aa, rr, r),
            lambda nn, aa, rr, r: random_config(nn, aa, rr, r),
        ]
        gfam = gobel_family(n, a, rng, rot)
        tfam = trump_family(n, a, rng, rot)
        bfam = bidwell_family(n, a, rng, rot)
        if gfam:
            inits.insert(0, lambda nn, aa, rr, r: gobel_config(
                nn, aa, rr, r, jitter=0.0))
        if tfam:
            inits.insert(0, lambda nn, aa, rr, r: trump_config(
                nn, aa, rr, r, jitter=0.0))
        if bfam:
            inits.insert(0, lambda nn, aa, rr, r: bidwell_config(
                nn, aa, rr, r, jitter=0.0))
        candidates.extend(gfam)
        candidates.extend(tfam)
        candidates.extend(bfam)
    elif c['cic']:
        inits = [
            lambda nn, aa, rr, r: grid_config(nn, aa, rr, jitter=0.02),
            lambda nn, aa, rr, r: ring_config(nn, aa, rr, r, rings=2),
            lambda nn, aa, rr, r: random_config(nn, aa, rr, r),
        ]
        if c['cis']:
            inits.insert(1, lambda nn, aa, rr, r: hex_config(nn, aa, rr, r))
    else:
        inits = [
            lambda nn, aa, rr, r: grid_config(nn, aa, rr, jitter=0.02),
            lambda nn, aa, rr, r: hex_config(nn, aa, rr, r),
            lambda nn, aa, rr, r: random_config(nn, aa, rr, r),
        ]
    if not expired():
        tasks = []
        for k in range(agents):
            base = inits[k % len(inits)](n, a, rng, rot)
            tasks.append((base[0], base[1], next_seed(), n, a, iters, rot,
                          opt, max_slots, de, H, arch))
        for (v, ca, th, bc, bth) in _pmap(_train_worker, tasks):
            if ca is not None:
                candidates.append((ca, th))
            # on garde aussi l'amorce brute (souvent déjà très bonne)
            candidates.append((bc, bth))

    # --- amorces structurées (A2/A6) : balayage des familles adaptées au
    #     mode (charpentes/palettes en carré ; grilles/anneaux sinon).
    #     Évaluation EXACTE gratuite de chaque membre ; on polit les 4
    #     meilleurs — c'est le polissage qui décide de la topologie.
    if not expired():
        if c['sqc'] and not c['cis']:
            fam = sweep_families(n, a, rng, rot)
        else:
            fam = generic_families(n, a, rng, rot)
        fam.sort(key=lambda z: exact_violation(z[0], z[1], a))
        ftop = fam[:4]
        fpol = _pmap(_polish_worker,
                     [(c, th, a, next_seed(), polish_iters, rot)
                      for (c, th) in ftop])
        for (v, c2, th2), (c, th) in zip(fpol, ftop):
            candidates.append((c2, th2))
            candidates.append((c, th))

    # --- amorces « trempe simulée » : exploration Monte-Carlo complémentaire
    #     du gradient (sauts entre bassins d'attraction), en parallèle :
    #     un départ aléatoire + un départ en grille.
    if anneal_iters > 0 and not expired():
        aruns = _pmap(_anneal_worker,
                      [(n, a, next_seed(), anneal_iters, rot, 0),
                       (n, a, next_seed(), anneal_iters, rot, 1)])
        for (v, c, th) in aruns:
            candidates.append((c, th))
        apol = _pmap(_polish_worker,
                     [(c, th, a, next_seed(), polish_iters, rot)
                      for (v, c, th) in aruns])
        for (v, c2, th2) in apol:
            candidates.append((c2, th2))

    if not candidates:
        candidates.append(grid_config(n, a, rng, jitter=0.0))

    # --- polissage des meilleurs candidats (indépendants -> parallèle)
    scored = []
    for (c, th) in candidates:
        scored.append((exact_violation(c, th, a), c, th))

    def _key(v, c, th):
        """Priorité : faisable d'abord, puis conteneur resserré le plus petit.

        À violation nulle, beaucoup de candidats sont valides mais de qualité
        très différente (une grille lâche comme une amorce structurée exacte).
        Sans ce départage, l'ordre d'insertion ferait perdre les meilleures
        structures (Göbel/Trump/Bidwell) au profit d'une grille valide."""
        if v <= FEAS_TOL:
            try:
                return (0.0, tighten_a(c, th, a)[0])
            except Exception:
                return (0.0, a)
        return (1.0, v)

    scored.sort(key=lambda z: _key(*z))
    best = scored[0] if scored else (float("inf"), None, None)
    best_key = _key(*best) if best[1] is not None else (float("inf"), 0.0)
    # --- A5 : sélection DIVERSE (écarte les quasi-doublons topologiques)
    if not expired():
        k_top = max(4, min(len(scored), max(3, len(scored) // 2)))
        top = diverse_top(scored, a, k_top)
        polished = _pmap(_polish_worker,
                         [(c, th, a, next_seed(), polish_iters, rot)
                          for (v0, c, th) in top])
        for (v, c2, th2) in polished:
            k2 = _key(v, c2, th2)
            if k2 < best_key:
                best, best_key = (v, c2, th2), k2

    # --- relances « coup de poing » (iterated local search) : on perturbe le
    #     meilleur candidat avec des amplitudes décroissantes et on re-polit,
    #     ce qui échappe aux minima locaux peu profonds du polissage.
    if kicks > 0 and best[1] is not None:
        for sigma in np.linspace(0.12, 0.03, kicks):
            if expired():
                break
            c2 = best[1] + rng.normal(0.0, sigma * a, best[1].shape)
            th2 = best[2] + (rng.normal(0.0, sigma, best[2].shape)
                             if rot else 0.0)
            v, c3, th3 = polish((c2, th2), a, polish_iters, rng, rot)
            k3 = _key(v, c3, th3)
            if k3 < best_key:
                best, best_key = (v, c3, th3), k3
    return best


# ===========================================================================
#  Estimation de C(n) : curricula + bissection
# ===========================================================================

def _attempt_worker(task):
    """Tâche parallèle (B3) : UN essai complet exécuté DANS un worker.

    Pose _IN_WORKER : les sous-tâches de l'essai sont alors exécutées
    séquentiellement sur CE worker — le pool du parent répartit les essais
    spéculatifs entre les cœurs au lieu de les embaucher tous en interne."""
    global _IN_WORKER
    (n, a, seed, kw) = task
    _IN_WORKER = True
    rr = np.random.default_rng(int(seed))
    return attempt(n, a, rr, **kw)


def estimate_C(n, rng, tol=2e-3, agents=6, iters=250, polish_iters=300,
               rot=True, opt="adam", max_slots=64, quiet=False,
               anneal_iters=1500, kicks=3, arch="set", spec=1,
               time_budget=0.0, bar_position=None, bar_leave=True,
               bar_desc=None):
    """Estime C(n) par curricula (descente de a) + bissection, avec CERTIF.

    À chaque essai on ne se fie pas seulement à la violation résiduelle de
    l'optimiseur : on certifie rigoureusement le résultat par
    rescale_to_feasible, qui fournit un empilement exactement valide à un
    côté a' >= a. La réponse est le plus petit a' certifié, majorant
    rigoureux de C(n).

    - B3 (`spec`) : la descente propose `spec` paliers sous le a courant,
      exécutés EN PARALLÈLE (essais indépendants = embarquement parfait).
      Si tous passent, on saute au plus profond et le pas grandit ; sinon on
      avance d'un seul pas.
    - A1 : un échec n'est jamais pris pour une preuve.  Il est confirmé par
      un essai plus profond au même `a` (démarrages chauds), puis par une
      sonde limitée sous la borne basse ; seule une réussite certifiée peut
      améliorer le majorant.
    - `time_budget` : budget par n en secondes (0 = illimité).  À l'expiration
      on renvoie le meilleur certificat déjà obtenu (le raffinement final est
      interrompu entre deux petits essais).
    - `bar_position`, `bar_leave`, `bar_desc` : contrôle de la barre tqdm
      (utile pour afficher la barre globale des runs et celle du run courant).
    """
    t0 = time.time()
    deadline = (t0 + float(time_budget)
                if time_budget and float(time_budget) > 0.0 else None)

    def hors_delai():
        return deadline is not None and time.time() >= deadline

    a_hi = initial_upper_bound(n)                   # grille : toujours faisable
    grid = grid_config(n, a_hi, rng, jitter=0.0)    # grille exacte (touchante)
    # certification par dilatation : le majorant initial est un vrai certificat
    a0, gc0, gth0 = certify(grid[0], grid[1], a_hi, rot)
    champs = [(gc0, gth0, a0)]
    best_a, best_cfg = a0, (gc0, gth0)             # plus petit côté CERTIFIÉ
    history = [(a0, 0.0)]

    log = (lambda *x: None) if quiet else print

    bar = None
    if not quiet:
        # Le nombre exact d'essais n'est pas connu à l'avance. On part d'une
        # estimation raisonnable ; la barre s'allonge si nécessaire, puis sa
        # borne est calée sur le nombre réel d'essais à la fin (100 %).
        bar_total = max(10, 4 + 2 * max(1, int(spec)))
        bar = tqdm(total=bar_total, desc=bar_desc or f"recherche C({n})",
                   unit="essai", dynamic_ncols=True, leave=bar_leave,
                   file=sys.stdout, position=bar_position)

        def log(*x):
            bar.write(" ".join(str(v) for v in x))

    log(f"  a départ (grille, faisable) : a = {a_hi:.6f}")

    def attempt_kw(light=False, deep=False):
        """Mot-clés d'attempt ; `light` = sonde spéculative (B3) : moins
        d'agents, ni trempe ni kicks — la détection de faisabilité vient du
        polissage, qui est conservé en entier.  `deep` = essai de confirmation
        d'un échec : plus d'agents, d'itérations, de trempe et de relances."""
        kw = dict(agents=agents, iters=iters, polish_iters=polish_iters,
                  champions=champs, rot=rot, opt=opt, max_slots=max_slots,
                  quiet=True, anneal_iters=anneal_iters, kicks=kicks,
                  arch=arch, deadline=deadline)
        if light:
            kw["agents"] = max(2, agents // 3)
            kw["anneal_iters"] = 0
            kw["kicks"] = 0
        if deep:
            kw["agents"] = max(agents + 4, int(1.6 * agents))
            kw["iters"] = int(1.5 * iters)
            kw["polish_iters"] = int(1.6 * polish_iters)
            kw["anneal_iters"] = int(1.5 * max(anneal_iters, 400))
            kw["kicks"] = kicks + 3
        return kw

    def _enregistre(a_test, v, c, th):
        """Certifie l'essai et met à jour le meilleur / les champions."""
        nonlocal best_a, best_cfg
        a_cert, cc, cth = certify(c, th, a_test, rot)
        history.append((a_test, v))
        if a_cert < best_a - 1e-12:
            best_a, best_cfg = a_cert, (cc, cth)
        champs.append((cc, cth, a_cert))
        if len(champs) > 14:            # borne le coût des essais suivants
            del champs[1:3]             # (champs[0] = grille, jamais virée)
        if bar is not None:
            if bar.n >= bar.total:
                # L'estimation initiale était trop basse : on allonge la
                # barre avant l'essai suivant, sans afficher un faux 100 %.
                bar.total = bar.n + max(5, 2 * max(1, int(spec)) + 1)
                bar.refresh()
            bar.update(1)
            bar.set_postfix_str(
                f"a={a_test:.4f} viol={v:.1e} best={best_a:.5f}")
        return a_cert

    durees = []

    def assez_de_temps(frac=0.5):
        """Évite de lancer un essai qui dépasserait nettement le budget."""
        if deadline is None:
            return True
        if not durees:
            return True
        est = sorted(durees)[len(durees) // 2]
        return (deadline - time.time()) >= frac * est

    def essai(a_test, light=False):
        """UN essai DANS le parent (sous-tâches réparties par le pool)."""
        t1 = time.time()
        v, c, th = attempt(n, a_test, rng, **attempt_kw(light))
        durees.append(time.time() - t1)
        return v, _enregistre(a_test, v, c, th)

    def essai_deep(a_test):
        """Essai de CONFIRMATION d'un échec, au même `a`.

        Un échec de l'optimiseur ne prouve pas l'infaisabilité : on relance
        avec davantage d'agents/itérations/trempe/relances avant de rabaisser
        la borne basse.  Retourne (v, a_certifié) ou None si le temps est
        écoulé."""
        if hors_delai() or not assez_de_temps(0.6):
            return None
        t1 = time.time()
        v, c, th = attempt(n, a_test, rng, **attempt_kw(deep=True))
        durees.append(time.time() - t1)
        return v, _enregistre(a_test, v, c, th)

    def essai_batch(rungs, light=False):
        """B3 — K essais spéculatifs en parallèle (pool), sinon 1 par 1.

        Repli séquentiel : on s'arrête au premier échec (pas de travail
        gratuit sous la borne). Retourne (v, a_cert) dans l'ordre des rungs."""
        if spec <= 1 or len(rungs) <= 1 or _POOL is None or _IN_WORKER:
            out = []
            for ar in rungs:
                out.append(essai(ar, light=False))
                if out[-1][0] > FEAS_TOL:
                    break
            return out
        kw = attempt_kw(light)
        tasks = [(n, ar, int(rng.integers(0, 2 ** 62)), kw) for ar in rungs]
        t1 = time.time()
        out = []
        for ar, (v, c, th) in zip(rungs, _pmap(_attempt_worker, tasks)):
            out.append((v, _enregistre(ar, v, c, th)))
        durees.append(time.time() - t1)
        return out

    # --- sonde ciblée n = k²+1 : la borne d'acceptation k + √2/2 est un
    #     objectif naturel. Pour n = 5 et 10, l'amorce EXACTE de Göbel y est
    #     faisable sans aucun polissage ; pour les autres k elle sert de
    #     point de départ direct, souvent meilleur que la grille k+1.
    if not _CIRC_SHAPE and not _CIRC_CONT:
        bnd = bound_k2_plus1(n)
        if bnd is not None and bnd < a_hi - 1e-12 and assez_de_temps(0.5):
            v_b, a_b = essai(bnd)
            log(f"  sonde n=k²+1 : a = {bnd:.6f}  viol = {v_b:.3e}  "
                f"certifié a' = {a_b:.6f}  "
                f"{'OK' if v_b <= FEAS_TOL else 'non'}")
            if v_b <= FEAS_TOL:
                a_hi = bnd

    # --- descente géométrique AVEC SAUTS SPÉCULATIFS (B3) : on propose
    #     `spec` paliers sous le a courant en parallèle ; si TOUS sont
    #     faisables on va au plus profond et le pas grandit (x1.5), sinon on
    #     avance d'un seul pas ou, au premier échec, on fixe la borne basse.
    a_lo = None
    a = a_hi
    step = max(0.02, 0.012 * a_hi)
    for step_i in range(80):
        if hors_delai() or not assez_de_temps(0.5):
            break
        rungs = [a - (k + 1) * step for k in range(max(1, spec))]
        rungs = [r for r in rungs
                 if r > area_lower_bound(n) - 1e-9]   # borne d'aire
        if not rungs:
            break
        results = essai_batch(rungs, light=(len(rungs) > 1))
        ok0 = results[0][0] <= FEAS_TOL
        if len(rungs) == 1:
            log(f"  descente #{step_i+1}: a = {rungs[0]:.6f}  "
                f"viol = {results[0][0]:.3e}  "
                f"certifié a' = {results[0][1]:.6f}  "
                f"{'OK' if ok0 else 'in faisable'}")
        else:
            desc = "   ".join(f"{ar:.4f}:{'OK' if v <= FEAS_TOL else 'non'}"
                              for ar, (v, _) in zip(rungs, results))
            log(f"  descente #{step_i+1} (spéculatif x{len(rungs)}): {desc}")
        if not ok0:
            # --- A1 : un échec n'est pas une preuve.  On confirme par un
            #     essai plus profond au MÊME a (démarrages chauds inclus)
            #     avant de rabaisser la borne basse.
            deep_ok = False
            d = essai_deep(rungs[0])
            if d is not None:
                deep_ok = d[0] <= FEAS_TOL
                log(f"  confirmation (approfondie) : a = {rungs[0]:.6f}  "
                    f"viol = {d[0]:.3e}  certifié a' = {d[1]:.6f}  "
                    f"{'OK' if deep_ok else 'échec confirmé'}")
            if deep_ok:
                a = rungs[0]
                continue
            # --- A1 bis : sondes limitées SOUS la borne basse.  Un succès
            #     certifié y est parfaitement valide (il ne sert qu'à sauter
            #     un bassin d'attraction) ; sinon on ne change pas la borne.
            probe_ok = False
            for frac in (0.01,):
                if hors_delai():
                    break
                ap = rungs[0] * (1.0 - frac)
                if ap <= area_lower_bound(n) + 1e-9:
                    break
                dp = essai_deep(ap)
                if dp is None:
                    break
                log(f"  sonde basse : a = {ap:.6f}  viol = {dp[0]:.3e}  "
                    f"certifié a' = {dp[1]:.6f}  "
                    f"{'OK' if dp[0] <= FEAS_TOL else 'non'}")
                if dp[0] <= FEAS_TOL:
                    probe_ok = True
                    a = ap
                    break
            if probe_ok:
                continue
            a_lo = rungs[0]             # échec confirmé -> borne basse
            break
        if all(v <= FEAS_TOL for (v, _) in results):
            a = rungs[-1]               # tout passe : saut au plus profond
            step *= 1.5
        else:
            a = rungs[0]                # avancée d'un pas seulement
    a_hi = a                            # dernier a où l'essai a réussi

    # --- bissection sur la faisabilité (a -> l'optimiseur y arrive-t-il ?)
    if a_lo is None:
        a_lo = area_lower_bound(n)      # borne d'aire : strictement en dessous,
                                        # l'empilement est impossible (non testé)
    while a_hi - a_lo > tol and not hors_delai() and assez_de_temps(0.5):
        am = 0.5 * (a_hi + a_lo)
        v, a_cert = essai(am)
        ok = v <= FEAS_TOL
        log(f"  bissection: a = {am:.6f}  viol = {v:.3e}  "
            f"certifié a' = {a_cert:.6f}  {'faisable' if ok else 'non'}")
        if ok:
            a_hi = am
        else:
            a_lo = am

    # --- A1 : raffinement « à pelures » sous la tolérance de bissection
    #     (sertissement + re-polissage dans un conteneur plus serré)
    best_a, _bc, _bt = peel_refine(best_cfg[0], best_cfg[1], best_a, rng, rot,
                                   deadline=deadline)
    best_cfg = (_bc, _bt)

    dt = time.time() - t0
    if deadline is not None and time.time() >= deadline:
        log("  budget de temps écoulé : meilleur certificat conservé")
    log(f"  -> C({n}) <= {best_a:.6f}   (certifié, {dt:.1f} s)")
    if bar is not None:
        # La recherche est finie : on cale la borne sur le nombre réel
        # d'essais pour que la barre affiche bien 100 %.
        if bar.n > 0:
            bar.total = bar.n
        bar.refresh()
        bar.close()
    return best_a, best_cfg, history, dt


# ===========================================================================
#  Tracé
# ===========================================================================

def plot_packing(centers, thetas, a, path, n, extra=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    c = _ctx(); cmap = plt.get_cmap("tab20")
    from matplotlib.patches import Circle as MC
    if c['sqc']:
        ax.add_patch(plt.Rectangle((0, 0), a, a, fill=False, lw=2, ec="black"))
        lim = (-0.08 * a, 1.08 * a)
    else:
        ax.add_patch(MC((0,0), a, fill=False, lw=2, ec="black"))
        lim = (-1.08*a, 1.08*a)
    if c['sqs']:
        off = np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]])
        for i in range(n):
            cc, th = centers[i], thetas[i]
            ct,st = math.cos(th), math.sin(th)
            Rm = np.array([[ct,-st],[st,ct]])
            V = cc + off @ Rm.T
            ax.add_patch(MplPolygon(V, closed=True, fc=cmap(i%20), ec="black", alpha=0.85, lw=0.8))
            ax.text(cc[0],cc[1],str(i+1),ha="center",va="center",fontsize=8)
        if c['sqc']:
            ax.set_title(f"n = {n} carrés unitaires dans un carré de côté a ≈ {a:.5f}\n"
                         f"densité n/a² = {n/a**2:.4f}  {extra}")
        else:
            ax.set_title(f"n = {n} carrés unitaires dans un cercle de rayon R ≈ {a:.5f}\n"
                         f"aire n/(π R²) = {n/(math.pi*a*a):.4f}  {extra}")
    else:
        for i in range(n):
            cc, th = centers[i], thetas[i]
            ax.add_patch(MC((cc[0],cc[1]), c['r'], fc=cmap(i%20), ec="black", alpha=0.85, lw=0.8))
            ax.text(cc[0],cc[1],str(i+1),ha="center",va="center",fontsize=8)
        if c['cic']:
            ax.set_title(f"n = {n} cercles unité (r={c['r']}) dans un cercle de rayon R ≈ {a:.5f}\n"
                         f"densité {(n*math.pi*c['r']*c['r'])/(math.pi*a*a):.4f}  {extra}")
        else:
            ax.set_title(f"n = {n} cercles unité (r={c['r']}) dans un carré de côté a ≈ {a:.5f}\n"
                         f"densité {(n*math.pi*c['r']*c['r'])/(a*a):.4f}  {extra}")
    ax.set_xlim(lim[0],lim[1]); ax.set_ylim(lim[0],lim[1]); ax.set_aspect("equal")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ===========================================================================
#  Vérification des gradients (gradcheck)
# ===========================================================================

def gradcheck(seed=0, archs=("set",)):
    """Vérifie par différences finies (1) le gradient de la perte géométrique
    et (2) la rétropropagation de l'agent Deep-Set."""
    rng = np.random.default_rng(seed)
    n, a = 5, 3.2
    centers = rng.uniform(0, a, (n, 2))
    thetas = rng.uniform(0, math.pi / 4, n)
    tau = 0.02
    L, dc, dt, _ = loss_and_backward(centers, thetas, a, tau)
    eps = 1e-6
    err = 0.0
    for i in range(n):
        for k in range(2):
            Cp = centers.copy(); Cp[i, k] += eps
            Cm = centers.copy(); Cm[i, k] -= eps
            Lp, _, _, _ = loss_and_backward(Cp, thetas, a, tau)
            Lm, _, _, _ = loss_and_backward(Cm, thetas, a, tau)
            num = (Lp - Lm) / (2 * eps)
            err = max(err, abs(num - dc[i, k]) / max(1.0, abs(num)))
    for i in range(n):
        Tp = thetas.copy(); Tp[i] += eps
        Tm = thetas.copy(); Tm[i] -= eps
        Lp, _, _, _ = loss_and_backward(centers, Tp, a, tau)
        Lm, _, _, _ = loss_and_backward(centers, Tm, a, tau)
        num = (Lp - Lm) / (2 * eps)
        err = max(err, abs(num - dt[i]) / max(1.0, abs(num)))
    print(f"gradcheck[perte] : erreur relative max = {err:.3e}  "
          f"{'OK' if err < 1e-4 else 'ATTENTION'}")
    global_err = err

    # gradients du réseau — une boucle par architecture
    de, H = 8, 16
    for arch in archs:
        params = init_agent(rng, 16, de, H, arch)
        base = grid_config(n, a, rng, jitter=0.05)
        X = build_inputs(params, n, a, base)
        c0, th0, cache, O = agent_config(params, X, base, True)
        _, dc, dt, _ = loss_and_backward(c0, th0, a, tau)
        O_grad = np.concatenate([dc, dt[:, None]], axis=1)
        grads, dX = set_backward(params, cache, O_grad)
        dE = np.zeros_like(params["E"])
        dE[:n] = dX[:, :de]
        grads["E"] = dE
        vec = ravel_params(params)

        def net_loss(vec, params=params, base=base):
            p = unravel_params(vec, params)
            # X dépend des embeddings : il faut le reconstruire à chaque appel
            Xp = build_inputs(p, n, a, base)
            c, th, _, _ = agent_config(p, Xp, base, True)
            Lk, _, _, _ = loss_and_backward(c, th, a, tau)
            return Lk

        offsets, off = {}, 0
        for kk in sorted(params):
            offsets[kk] = off
            off += params[kk].size

        err_a = 0.0
        for k in sorted(grads):
            shape = grads[k].shape
            base_off = offsets[k]
            it = np.nditer(grads[k], flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                j = base_off + int(np.ravel_multi_index(idx, shape))
                vp = vec.copy(); vp[j] += eps
                vm = vec.copy(); vm[j] -= eps
                num = (net_loss(vp) - net_loss(vm)) / (2 * eps)
                err_a = max(err_a, abs(num - grads[k][idx])
                            / max(1.0, abs(num)))
                it.iternext()
        print(f"gradcheck[arch={arch}] : erreur relative max = {err_a:.3e}  "
              f"{'OK' if err_a < 1e-4 else 'ATTENTION'}")
        global_err = max(global_err, err_a)
    print(f"gradcheck : erreur relative max = {global_err:.3e}  "
          f"{'OK' if global_err < 1e-4 else 'ATTENTION'}")
    return global_err < 1e-4


# ===========================================================================
#  Benchmarck vs Wikipédia
# ===========================================================================



def bench(nmin=1, nmax=10, seed=0, agents=6, iters=250, polish_iters=300,
          rot=True, opt="adam", tol=2e-3, figs=True,
          anneal_iters=1500, kicks=3,
          outdir=DEFAULT_OUTDIR, arch="set", spec=1, time_budget=0.0):
    print("=" * 78)
    print(f"BENCHMARK — C(n) pour n = {nmin}..{nmax} "
          f"(mode {PACK})")
    print("=" * 78)
    print(f"{'n':>3} {'a trouvé':>12} {'densité':>9} {'temps':>8}  état")
    n_ok = 0
    it = tqdm(range(nmin, nmax + 1), desc=f"bench {nmin}→{nmax}",
              unit="n", dynamic_ncols=True, leave=True, file=sys.stdout)
    for n in it:
        rng = np.random.default_rng(seed + n)
        a_hat, (c, th), hist, dt = estimate_C(
            n, rng, tol=tol, agents=agents, iters=iters,
            polish_iters=polish_iters, rot=rot, opt=opt, quiet=True,
            anneal_iters=anneal_iters, kicks=kicks,
            arch=arch, spec=spec, time_budget=time_budget)
        # --- règles : solution sur grille OU n = k²+1 au-dessus de la borne
        #     k + √2/2 -> RIEN enregistré, RIEN généré
        ok_sol, raison_sol = solution_acceptable(n, a_hat)
        if not ok_sol:
            it.write(f"  n = {n:>3} : refusé — {raison_sol}")
            it.set_postfix_str(f"n={n} refusé")
            continue
        n_ok += 1
        it.write(f"  n = {n:>3} : a = {a_hat:.6f}   "
                 f"densité = {density(n, a_hat):.4f}   ({dt:.1f} s)")
        if figs:
            os.makedirs(dir_figures(outdir), exist_ok=True)
            plot_packing(c, th, a_hat, figure_path(n, outdir), n)
    it.close()
    print("=" * 78)
    if n_ok:
        print(f"{n_ok} solution(s) acceptée(s) — figures dans "
              f"{dir_figures(outdir)}/empilement_n*_*.png")
    else:
        print("aucune solution acceptée : rien ni enregistré ni généré.")


# ===========================================================================
#  CLI
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Empilement optimal avec IA : "
                    "1) carré dans carré  2) cercle dans cercle  "
                    "3) cercle dans carré  4) carré dans cercle")
    ap.add_argument("-n", "--n", type=int, default=5,
                    help="nombre d'objets unitaires (défaut : 5)")
    ap.add_argument("--pack", type=str, default="sq_in_sq",
                    choices=["sq_in_sq", "1", "cercle_dans_cercle", "circ_in_circ", "2",
                             "cercle_dans_carre", "circ_in_sq", "3",
                             "carre_dans_cercle", "sq_in_circ", "4"],
                    help="type d'empilement : 1=sq_in_sq (défaut), "
                         "2=circ_in_circ, 3=circ_in_sq, 4=sq_in_circ")
    ap.add_argument("--iters", type=int, default=250,
                    help="itérations d'entraînement réseau par agent")
    ap.add_argument("--agents", type=int, default=6,
                    help="nombre d'agents (population) entraînés")
    ap.add_argument("--polish", type=int, default=300,
                    help="itérations de polissage des coordonnées")
    ap.add_argument("--opt", choices=["adam", "es"], default="adam",
                    help="optimiseur de l'agent : adam (défaut) ou es")
    ap.add_argument("--no-rotate", action="store_true",
                    help="interdire la rotation des carrés (alignés axe)")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="tolérance de bissection sur a (défaut : 2e-3)")
    ap.add_argument("--anneal-iters", type=int, default=1500,
                    help="itérations de trempe simulée par essai (0 = off)")
    ap.add_argument("--kicks", type=int, default=3,
                    help="relances « coup de poing » (iterated local search)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processus parallèles (agents + polissages) ; "
                         "1 = séquentiel. Défaut : tous les cœurs")
    ap.add_argument("--max-slots", type=int, default=64,
                    help="taille minimale de la table d'embeddings "
                         "(étendue automatiquement à n si n est plus grand)")
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                    help="dossier de sortie (figures/) ; défaut : "
                         f"{DEFAULT_OUTDIR!r} (env EMPILEMENT_OUTDIR)")
    ap.add_argument("--spec", type=int, default=1,
                    help="essais spéculatifs parallèles pendant la descente "
                         "(B3) ; 1 = descente séquentielle (DÉFAUT — mesuré : "
                         "gain de temps nul et frontière légèrement "
                         "conservative avec K>1 sur n=26), K>1 = sondes "
                         "légères groupées (expérimental)")
    ap.add_argument("--runs", type=int, default=1,
                    help="runs indépendants multi-graines, exécutés l'un "
                         "après l'autre (barre globale + barre du run "
                         "courant) ; le meilleur résultat est conservé")
    ap.add_argument("--time-budget", type=float, default=0.0, metavar="SEC",
                    help="budget de temps PAR n en secondes (0 = illimité). "
                         "À l'expiration, le meilleur empilement déjà certifié "
                         "est renvoyé ; le raffinement final est interrompu "
                         "proprement.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="fichier PNG du résultat (défaut : "
                         "resultats/figures/empilement_n<n>_<mode>.png)")
    ap.add_argument("--no-fig", action="store_true", help="ne rien tracer")
    ap.add_argument("--range", type=str, default=None, metavar="A;B",
                    help="benchmark de n = A à B (ex. --range '11;15'). "
                         "Séparateurs acceptés : ; : , .. ou - ; le ';' doit "
                         "être protégé du shell (guillemets)")
    ap.add_argument("--gradcheck", action="store_true",
                    help="vérifier les gradients analytiques (diff. finies)")
    args = ap.parse_args(argv)

    # mapping pack
    pm = {"1": "sq_in_sq", "cercle_dans_cercle": "circ_in_circ", "2": "circ_in_circ",
          "cercle_dans_carre": "circ_in_sq", "3": "circ_in_sq",
          "carre_dans_cercle": "sq_in_circ", "4": "sq_in_circ"}
    set_pack(pm.get(str(args.pack), str(args.pack)))

    if args.gradcheck:
        ok = gradcheck()
        return 0 if ok else 1

    rot = not args.no_rotate
    warmup_jit()                # B2 : compilation JIT (et scipy) AVANT fork
    setup_pool(args.workers)    # pool de processus (agents + polissages)
    # B3 : au plus un essai spéculatif par cœur (sinon les vagues s'empilent)
    spec = max(1, min(int(args.spec), int(args.workers)))
    outdir = args.outdir

    if args.range is not None:
        vals = [v.strip() for v in
                args.range.replace("..", ";").replace(",", ";")
                .replace(":", ";").replace("-", ";").split(";")
                if v.strip()]
        if len(vals) == 1:
            nmin = nmax = int(vals[0])
        elif len(vals) == 2:
            nmin, nmax = int(vals[0]), int(vals[1])
        elif len(vals) == 0:
            ap.error("--range attend la forme A;B")
        else:
            ap.error("--range attend au plus deux bornes : A;B")
        if nmin > nmax:
            nmin, nmax = nmax, nmin
        if nmin < 1:
            ap.error("--range : A >= 1")
        bench(nmin=nmin, nmax=nmax, seed=args.seed,
              agents=args.agents, iters=args.iters, polish_iters=args.polish,
              rot=rot, opt=args.opt, tol=args.tol, figs=not args.no_fig,
              anneal_iters=args.anneal_iters, kicks=args.kicks,
              outdir=outdir, arch="set", spec=spec,
              time_budget=args.time_budget)
        return 0

    n = args.n
    if n < 1:
        ap.error("n >= 1")

    rng = np.random.default_rng(args.seed)

    est_kw = dict(tol=args.tol, agents=args.agents, iters=args.iters,
                  polish_iters=args.polish, rot=rot, opt=args.opt,
                  max_slots=args.max_slots, anneal_iters=args.anneal_iters,
                  kicks=args.kicks, arch="set", time_budget=args.time_budget)
    if args.runs > 1:
        # B4 : runs indépendants multi-graines.  Ils sont exécutés l'un
        # après l'autre pour que la barre du run courant reste lisible ;
        # le parallélisme interne (agents, polissages, --workers) reste actif.
        bar_runs = tqdm(total=args.runs, desc=f"runs C({n})", unit="run",
                        dynamic_ncols=True, leave=True, file=sys.stdout,
                        position=0)
        results = []
        try:
            for r in range(args.runs):
                rr = np.random.default_rng(args.seed + 977 * r)
                res = estimate_C(
                    n, rr, **est_kw,
                    bar_desc=f"run {r + 1}/{args.runs} C({n})",
                    bar_position=1, bar_leave=False)
                results.append(res)
                best_so_far = min(results, key=lambda t: t[0])[0]
                bar_runs.set_postfix_str(f"meilleur={best_so_far:.6f}")
                bar_runs.update(1)
        finally:
            bar_runs.close()
        for i, (ra, _rc, _rh, rdt) in enumerate(results):
            print(f"  run {i + 1}/{args.runs} : C({n}) <= {ra:.6f}  "
                  f"({rdt:.1f} s)")
        a_hat, (c, th), hist, dt = min(results, key=lambda t: t[0])
        print(f"(meilleur des {args.runs} runs retenu)")
    else:
        a_hat, (c, th), hist, dt = estimate_C(n, rng, **est_kw)

    print("-" * 74)
    print(f"meilleure taille de conteneur trouvée : a ≈ {a_hat:.6f}   "
          f"densité = {density(n, a_hat):.4f}   ({dt:.1f} s)")
    v = exact_violation(c, th, a_hat)
    print(f"vérification exacte (géométrie interne) : violation = {v:.3e} "
          f"-> {'valide' if v <= FEAS_TOL else 'INVALIDE'}")
    # --- règles d'enregistrement : grille (a entier) ou n = k²+1 au-dessus
    #     de la borne k + √2/2 -> RIEN enregistré, RIEN généré
    ok_sol, raison_sol = solution_acceptable(n, a_hat)
    if not ok_sol:
        print(f"solution refusée : {raison_sol}")
        return 0
    # --- sauvegarde texte/dossier supprimés : seule la figure individuelle ---
    if not args.no_fig:
        os.makedirs(dir_figures(outdir), exist_ok=True)
        path = args.out or figure_path(n, outdir)
        plot_packing(c, th, a_hat, path, n)
        print(f"figure : {path}")
    else:
        print("(pas de figure demandée)")
    print("positions (x, y, theta) :")
    for i in range(n):
        print(f"  {i+1:3d}  {c[i,0]: .6f} {c[i,1]: .6f} {th[i]: .6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
