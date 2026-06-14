# -*- coding: utf-8 -*-
"""
Capítulo 7 — Estudio de simulación Monte Carlo según
`Capitulo7_Protocolo_Simulacion.docx` (9 pasos del protocolo del director).

Etapas (flags): --calibrar  --piloto  --rejilla  --productos  --validar
(ver resultados/cap7_preregistro.md para el preregistro completo).

Generador (Pasos 1-3):
  - Bifactor a nivel de ítem con las cargas publicadas (Cuesta et al., 2018,
    Tabla 2). Las cargas NO se tocan.
  - Umbrales Likert calibrados POR DIMENSIÓN (desplazamiento a_k y escala
    b_k sobre una base común) para que la población general del simulador
    reproduzca las normas de empleados de Cuesta Tabla 7 (medias y SD).
    La saturación (Pc>=95) se evalúa contra esa población normativa interna.
  - Grit cargado en g y en el específico de logro (pesos calibrados para
    r(grit, ML) ~= 0,61) con mapeo afín calibrado a la media/SD reales.
  - Efecto verdadero en g: P(y=1) = enlace(b0 + b1*g); d definido en la
    POBLACIÓN GENERAL (verificado por esperanzas en megamuestra 10^6);
    b0 calibrado a tasa base 0,634 en la muestra seleccionada.
  - Preselección: DESVIACIÓN REGISTRADA — selección top-tau sobre un
    compuesto ruidoso v = rho*z(suma BEPE) + sqrt(1-rho^2)*eps, porque el
    truncamiento duro sobre la suma observada es incompatible con la
    muestra real (desplazada +0,9 SD pero con U de Thorndike ~1 e
    intercorrelación intacta). rho y tau* se calibran juntos en el Paso 4.

Criterio de detección único: AUC_cv > p95 de la celda gemela d=0 (mismas
n y tau). Pipeline por réplica = Cap. 5: elastic-net (C x l1_ratio, saga),
9 predictores, CV anidada (inner 5-fold); outer 5-fold estratificado en la
rejilla (LOO inviable: ver --piloto; desviación registrada).

Semillas: SeedSequence raíz 20260611, spawn jerárquico, registradas.
Salidas en resultados/ con prefijo cap7_.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------------------------------------------------------------- constantes
SEED_RAIZ = 20260611
TASA_BASE = 0.634
N_GRID = [41, 60, 80, 120, 160, 200, 300, 500]
D_GRID = [0.0, 0.2, 0.35, 0.5, 0.8]
M_REP = 500
MIN_MINORIA = 8
N_MEGA = 1_000_000
N_POOL_CAL = 150_000

BASE_DIR = Path(__file__).resolve().parents[1]
DATA = BASE_DIR / 'data' / 'dataset_franquicias_seudonimizado.xlsx'
OUT = BASE_DIR / 'resultados'

# Cargas publicadas (Cuesta et al., 2018, Tabla 2) — NO se tocan (Paso 4).
LG = {'SE': [.72, .75, .71, .69, .70, .79, .65, .72, .78, .70],
      'AU': [.43, .14, .61, .63, .32, .26, .30, .52, .55, .56],
      'IN': [.48, .62, .56, .63, .48, .56, .60, .74, .58, .56],
      'IL': [.47, .63, .50, .52, .42, .31, .60, .39, .43, .46],
      'AM': [.59, .67, .61, .52, .67, .70, .60, .69, .66, .58],
      'OP': [.59, .55, .56, .62, .79, .62, .50, .51, .71, .48],
      'ST': [.72, .27, .54, .37, .46, .42, .38, .42, .36, .53],
      'RT': [.61, .73, .50, .56, .66, .59, .57, .64, .56, .46]}
LS = {'SE': [-.02, -.28, .54, -.15, .24, -.02, .49, .12, .11, -.26],
      'AU': [.47, .61, .18, .02, .70, .66, .67, .52, .36, .21],
      'IN': [.48, .49, .30, .45, .32, .49, .49, .35, .41, .41],
      'IL': [.69, .40, .61, .30, .44, .43, .24, .64, .55, .66],
      'AM': [.40, .37, .34, .37, -.07, .05, .26, .06, .46, .58],
      'OP': [.57, .56, .39, .45, .31, .44, .58, .47, .19, .61],
      'ST': [.20, .47, .41, .42, .48, .74, .67, .47, .50, .23],
      'RT': [.39, .28, .55, .19, .21, .60, .14, .28, .18, .50]}
DIMS_PUB = list(LG.keys())                                  # SE AU IN IL AM OP ST RT
DIMS_BEPE = ['AE', 'AU', 'IN', 'LI', 'ML', 'OP', 'TE', 'TR']
LAM_G = np.concatenate([LG[k] for k in DIMS_PUB])
LAM_S = np.concatenate([LS[k] for k in DIMS_PUB])
RES_SD = np.sqrt(np.clip(1 - LAM_G**2 - LAM_S**2, 0.05, None))
DIM_IDX = np.repeat(np.arange(8), 10)
THR_BASE = np.array([-2.1, -1.3, -0.55, 0.55])
IDX_AM, IDX_ST = 4, 6

# Normas de empleados (Cuesta et al., 2018, Tabla 7): objetivo poblacional
NORMAS = {'AE': (37.55, 5.05), 'AU': (40.64, 4.98), 'IN': (38.75, 4.50),
          'LI': (39.54, 5.36), 'ML': (39.91, 4.43), 'OP': (38.77, 5.27),
          'TE': (32.66, 6.57), 'TR': (36.84, 5.31)}


# ---------------------------------------------------------------- generador
def latentes(n, rng, delta_g=0.0):
    """delta_g: desplazamiento del pool de aspirantes en el factor general
    (población emprendedora vs normas de empleados; ancla: Cuesta MANOVA)."""
    g = delta_g + rng.standard_normal(n)
    s = rng.standard_normal((n, 8))
    lat = np.outer(g, LAM_G) + s[:, DIM_IDX] * LAM_S + rng.standard_normal((n, 80)) * RES_SD
    return g, s, lat


def discretizar(lat, thr_dim):
    """Likert 1-5 con umbrales por dimensión; PD = suma de 10 ítems."""
    n = lat.shape[0]
    pd_ = np.empty((n, 8))
    for k in range(8):
        bloque = lat[:, k * 10:(k + 1) * 10]
        likert = 1 + (bloque[:, :, None] >= thr_dim[k][None, None, :]).sum(axis=2)
        pd_[:, k] = likert.sum(axis=1)
    return pd_


def gen_grit(g, s, rng, grit):
    glat = grit['w'] * g + grit['w'] * s[:, IDX_AM] + \
        rng.standard_normal(len(g)) * np.sqrt(max(1 - 2 * grit['w']**2, 0.01))
    return np.clip(np.round((glat * grit['escala'] + grit['centro']) * 2) / 2, 1, 5)


def enlace(eta, link='logit'):
    if link == 'probit':
        from scipy.stats import norm
        return norm.cdf(eta)
    return 1.0 / (1.0 + np.exp(-eta))


def d_general(b0, b1, g_mega, link='logit'):
    """d de g entre y=1/y=0 en población general, por esperanzas sobre la
    megamuestra (sin ruido Bernoulli)."""
    p = enlace(b0 + b1 * g_mega, link)
    w1 = p.mean()
    m1 = (g_mega * p).mean() / w1
    m0 = (g_mega * (1 - p)).mean() / (1 - w1)
    v1 = (g_mega**2 * p).mean() / w1 - m1**2
    v0 = (g_mega**2 * (1 - p)).mean() / (1 - w1) - m0**2
    sp = np.sqrt(w1 * v1 + (1 - w1) * v0)
    return (m1 - m0) / sp


def calibrar_celda(d_obj, g_sel, g_mega, link='logit', tasa=TASA_BASE):
    from scipy.optimize import brentq
    b0, b1 = 0.0, 0.0
    for _ in range(4):
        if d_obj > 0:
            b1 = brentq(lambda b: d_general(b0, b, g_mega, link) - d_obj, 1e-6, 8.0)
        b0 = brentq(lambda b: enlace(b + b1 * g_sel, link).mean() - tasa, -8, 8)
    return b0, b1


# ---------------------------------------------------------------- réplica
# Autocontenida: los workers de loky en Windows no resuelven globals de __main__.
def replica_cap7(n, tau, d_obj, b0, b1, semilla, outer, lam_g, lam_s, res_sd,
                 dim_idx, thr_dim, grit, rho, delta_g, delta_s, idx_am,
                 min_minoria, link='logit', efecto='general', idx_st=6,
                 delta_s2=None, peso_mezcla=None):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import (GridSearchCV, LeaveOneOut,
                                         StratifiedKFold, cross_val_predict)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import norm

    rng = np.random.default_rng(semilla)
    pool = int(np.ceil(n / tau)) if tau < 1 else n
    thr_dim = np.asarray(thr_dim)

    delta_s = np.asarray(delta_s)
    for _ in range(200):
        g = delta_g + rng.standard_normal(pool)
        if delta_s2 is None:
            desplaz = delta_s[None, :]
        else:                       # mezcla de dos enseñas (sensibilidad)
            comp = rng.random(pool) < peso_mezcla
            desplaz = np.where(comp[:, None], delta_s[None, :],
                               np.asarray(delta_s2)[None, :])
        s = desplaz + rng.standard_normal((pool, 8))
        lat = np.outer(g, lam_g) + s[:, dim_idx] * lam_s + \
            rng.standard_normal((pool, 80)) * res_sd
        pd_ = np.empty((pool, 8))
        for k in range(8):
            bloque = lat[:, k * 10:(k + 1) * 10]
            pd_[:, k] = (1 + (bloque[:, :, None] >= thr_dim[k][None, None, :])
                         .sum(axis=2)).sum(axis=1)
        glat = grit['w'] * g + grit['w'] * s[:, idx_am] + \
            rng.standard_normal(pool) * np.sqrt(max(1 - 2 * grit['w']**2, 0.01))
        ego = np.clip(np.round((glat * grit['escala'] + grit['centro']) * 2) / 2, 1, 5)
        if tau < 1:
            suma = pd_.sum(axis=1)
            v = rho * (suma - suma.mean()) / suma.std() + \
                np.sqrt(1 - rho**2) * rng.standard_normal(pool)
            idx = np.argsort(-v)[:n]
        else:
            idx = np.arange(n)
        x = np.column_stack([pd_[idx], ego[idx]])
        factor = g[idx] if efecto == 'general' else \
            (s[idx, idx_st] + s[idx, idx_am]) / np.sqrt(2)
        eta = b0 + b1 * factor
        p = norm.cdf(eta) if link == 'probit' else 1 / (1 + np.exp(-eta))
        y = (rng.random(n) < p).astype(int)
        if min(y.sum(), n - y.sum()) >= min_minoria:
            break
    else:
        raise RuntimeError(f'réplica inviable n={n} tau={tau} d={d_obj}')

    pipe = Pipeline([('sc', StandardScaler()),
                     ('lr', LogisticRegression(solver='saga', max_iter=5000))])
    grid = {'lr__C': [0.01, 0.1, 1.0], 'lr__l1_ratio': [0.0, 0.5, 1.0]}
    gs = GridSearchCV(pipe, grid, cv=5, scoring='roc_auc', error_score=np.nan)
    cv_out = LeaveOneOut() if outer == 'loo' else \
        StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(2**31)))
    proba = cross_val_predict(gs, x, y, cv=cv_out, method='predict_proba')[:, 1]
    auc_cv = roc_auc_score(y, proba)
    gs.fit(x, y)
    auc_resub = roc_auc_score(y, gs.predict_proba(x)[:, 1])
    coef = gs.best_estimator_[-1].coef_.ravel()
    return dict(n=n, tau=tau, d=d_obj,
                semilla=int(np.asarray(semilla.entropy).ravel()[0]) if hasattr(semilla, 'entropy') else -1,
                auc_cv=auc_cv, auc_resub=auc_resub,
                C=gs.best_params_['lr__C'], l1_ratio=gs.best_params_['lr__l1_ratio'],
                n_coef_no_cero=int((np.abs(coef) > 1e-8).sum()),
                tasa=float(y.mean()))


# Réplica del generador alternativo por cópula gaussiana (sensibilidad 8.2a).
# Autocontenida por la misma razón que replica_cap7.
def replica_copula(n, d_obj, b0, b1, semilla, chol, margenes, w1, min_minoria,
                   link='logit'):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import norm

    rng = np.random.default_rng(semilla)
    chol = np.asarray(chol)
    margenes = np.asarray(margenes)          # (9, 41) valores reales ordenados
    w1 = np.asarray(w1)
    probs = (np.arange(margenes.shape[1]) + 0.5) / margenes.shape[1]

    for _ in range(200):
        z = rng.standard_normal((n, 9)) @ chol.T
        f = z @ w1                            # factor latente, varianza 1
        u = norm.cdf(z)
        x = np.column_stack([np.interp(u[:, j], probs, margenes[j])
                             for j in range(9)])
        eta = b0 + b1 * f
        p = norm.cdf(eta) if link == 'probit' else 1 / (1 + np.exp(-eta))
        y = (rng.random(n) < p).astype(int)
        if min(y.sum(), n - y.sum()) >= min_minoria:
            break
    else:
        raise RuntimeError(f'réplica cópula inviable n={n} d={d_obj}')

    pipe = Pipeline([('sc', StandardScaler()),
                     ('lr', LogisticRegression(solver='saga', max_iter=5000))])
    grid = {'lr__C': [0.01, 0.1, 1.0], 'lr__l1_ratio': [0.0, 0.5, 1.0]}
    gs = GridSearchCV(pipe, grid, cv=5, scoring='roc_auc', error_score=np.nan)
    cv_out = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(2**31)))
    proba = cross_val_predict(gs, x, y, cv=cv_out, method='predict_proba')[:, 1]
    auc_cv = roc_auc_score(y, proba)
    gs.fit(x, y)
    return dict(n=n, tau=-1, d=d_obj, semilla=-1, auc_cv=auc_cv,
                auc_resub=roc_auc_score(y, gs.predict_proba(x)[:, 1]),
                C=gs.best_params_['lr__C'], l1_ratio=gs.best_params_['lr__l1_ratio'],
                n_coef_no_cero=int((np.abs(gs.best_estimator_[-1].coef_) > 1e-8).sum()),
                tasa=float(y.mean()))


# ---------------------------------------------------------------- calibración
def cargar_real():
    df = pd.read_excel(DATA).dropna(subset=['exito'])
    cols = [f'BEPE_{k}_PD' for k in DIMS_BEPE]
    x = df[cols].to_numpy(float)
    sat, cortes = [], []
    for k in DIMS_BEPE:
        alto = df[df[f'BEPE_{k}_Pc'] >= 95][f'BEPE_{k}_PD']
        bajo = df[df[f'BEPE_{k}_Pc'] < 95][f'BEPE_{k}_PD']
        sat.append(float((df[f'BEPE_{k}_Pc'] >= 95).mean() * 100))
        # corte de la tabla normativa: menor PD consistente con los datos
        # (la mayor PD observada con Pc<95, más 1; acotado por la mínima
        # PD observada con Pc>=95)
        cortes.append(float(min(bajo.max() + 1, alto.min())) if len(alto) else np.nan)
    iu = np.triu_indices(8, 1)
    return dict(medias=x.mean(0), sds=x.std(0, ddof=1), sat=np.array(sat),
                cortes_pc95=np.array(cortes),
                intercorr=float(np.corrcoef(x.T)[iu].mean()),
                grit_media=float(df['ego'].mean()), grit_sd=float(df['ego'].std(ddof=1)),
                grit_ml=float(df['ego'].corr(df['BEPE_ML_PD'])))


def calibrar_umbrales(lat_cal):
    """(a_k, b_k) por dimensión: thr_k = a_k + b_k*THR_BASE, de modo que la
    población general reproduzca las normas de Cuesta Tabla 7 (media y SD)."""
    a = np.zeros(8)
    b = np.ones(8)
    for _ in range(25):
        thr_dim = a[:, None] + b[:, None] * THR_BASE[None, :]
        pd_ = discretizar(lat_cal, thr_dim)
        for k, dim in enumerate(DIMS_BEPE):
            m_obj, s_obj = NORMAS[dim]
            a[k] += 0.04 * (pd_[:, k].mean() - m_obj)       # subir umbral baja la media
            b[k] *= (pd_[:, k].std(ddof=1) / s_obj) ** 0.5  # estrechar umbral sube la SD
    return a[:, None] + b[:, None] * THR_BASE[None, :]


def calibrar(args):
    OUT.mkdir(exist_ok=True)
    real = cargar_real()
    ss = np.random.SeedSequence(SEED_RAIZ)
    s_cal, s_mega, s_grit = ss.spawn(3)
    rng = np.random.default_rng(s_cal)
    t0 = time.perf_counter()

    g_pop, s_pop, lat_pop = latentes(N_POOL_CAL, rng)
    thr_dim = calibrar_umbrales(lat_pop)
    pd_pop = discretizar(lat_pop, thr_dim)
    p95 = np.percentile(pd_pop, 95, axis=0)        # percentil interno (diagnóstico)
    # cortes de saturación: los de la tabla normativa real (recuperados del
    # dataset: PD mínima con Pc>=95); verificable contra p95 interno
    cortes = cargar_real()['cortes_pc95']

    fila_norm = [dict(dim=d, metrica='media_poblacional', real=NORMAS[d][0],
                      sintetico=pd_pop[:, k].mean(),
                      error_pct=100 * (pd_pop[:, k].mean() - NORMAS[d][0]) / NORMAS[d][0])
                 for k, d in enumerate(DIMS_BEPE)]
    fila_norm += [dict(dim=d, metrica='sd_poblacional', real=NORMAS[d][1],
                       sintetico=pd_pop[:, k].std(ddof=1),
                       error_pct=100 * (pd_pop[:, k].std(ddof=1) - NORMAS[d][1]) / NORMAS[d][1])
                  for k, d in enumerate(DIMS_BEPE)]

    # ---- pool emprendedor + selección ruidosa: búsqueda (delta_g, tau, rho)
    rng_g = np.random.default_rng(s_grit)
    eps = rng_g.standard_normal(N_POOL_CAL)
    iu = np.triu_indices(8, 1)
    sat_real, medias_r, sds_r = real['sat'], real['medias'], real['sds']

    def pool_con_delta(delta, delta_s=None):
        """Reutiliza los mismos latentes desplazando g y, opcionalmente, los
        específicos (perfil del pool de candidatos a franquicia)."""
        lat = lat_pop + delta * LAM_G[None, :]
        if delta_s is not None:
            lat = lat + delta_s[DIM_IDX] * LAM_S[None, :]
        return discretizar(lat, thr_dim)

    def metricas_sel(pd_d, tau, rho):
        suma = pd_d.sum(axis=1)
        v = rho * (suma - suma.mean()) / suma.std() + np.sqrt(1 - rho**2) * eps
        sel = np.argsort(-v)[:max(int(N_POOL_CAL * tau), 100)] if tau < 1 \
            else np.arange(N_POOL_CAL)
        pp = pd_d[sel]
        return (sel, pp.mean(0), pp.std(0, ddof=1),
                np.array([(pp[:, k] >= cortes[k]).mean() * 100 for k in range(8)]),
                float(np.corrcoef(pp.T)[iu].mean()))

    def error_total(m_s, sd_s, sat_s, ic_s):
        return (((m_s - medias_r) / medias_r)**2).sum() \
            + (((sd_s - sds_r) / sds_r)**2).sum() \
            + (((sat_s - sat_real) / np.maximum(sat_real, 10))**2).sum() \
            + ((ic_s - real['intercorr']) / real['intercorr'])**2 * 4

    def busca_tau_rho(pd_d):
        mejor = None
        for tau in np.round(np.arange(0.1, 0.95, 0.05), 2):   # τ* < 1 (hubo selección)
            for rho in np.round(np.arange(0.2, 1.01, 0.1), 2):
                _, m_s, sd_s, sat_s, ic_s = metricas_sel(pd_d, tau, rho)
                err = error_total(m_s, sd_s, sat_s, ic_s)
                if mejor is None or err < mejor[0]:
                    mejor = (err, tau, rho)
        return mejor

    def ajusta_delta_s(delta, tau, rho, delta_s):
        """Desplazamientos residuales por específico para clavar las 8 medias
        de la muestra seleccionada (10 iteraciones amortiguadas)."""
        for _ in range(10):
            pd_d = pool_con_delta(delta, delta_s)
            _, m_s, *_ = metricas_sel(pd_d, tau, rho)
            delta_s = delta_s + 0.25 * (medias_r - m_s)
        return delta_s

    # etapa 1: (δ_g, τ, ρ) sin residuales; etapas 2-4: alternar δ_k y (τ, ρ)
    mejor = None
    for delta in np.round(np.arange(0.0, 1.30, 0.25), 2):
        err, tau, rho = busca_tau_rho(pool_con_delta(delta))
        if mejor is None or err < mejor[0]:
            mejor = (err, delta, tau, rho)
    _, delta_g, tau_star, rho_star = mejor
    delta_s = np.zeros(8)
    for _ in range(2):
        delta_s = ajusta_delta_s(delta_g, tau_star, rho_star, delta_s)
        _, tau_star, rho_star = busca_tau_rho(pool_con_delta(delta_g, delta_s))

    pd_pool = pool_con_delta(delta_g, delta_s)
    g_pool = g_pop + delta_g
    s_pool = s_pop + delta_s[None, :]
    sel, m_s, sd_s, sat_s, ic_s = metricas_sel(pd_pool, tau_star, rho_star)

    # ---- grit: w por r(grit,ML) y (centro, escala) por media/SD, en seleccionados
    mejor_g = None
    for w in [0.40, 0.45, 0.50, 0.55, 0.60]:
        glat = w * g_pool + w * s_pool[:, IDX_AM] + \
            rng_g.standard_normal(N_POOL_CAL) * np.sqrt(max(1 - 2 * w**2, 0.01))
        for centro in np.arange(3.5, 4.65, 0.05):
            for escala in np.arange(0.30, 0.85, 0.05):
                ego = np.clip(np.round((glat[sel] * escala + centro) * 2) / 2, 1, 5)
                r_gml = np.corrcoef(ego, pd_pool[sel, IDX_AM])[0, 1]
                err = ((ego.mean() - real['grit_media']) / real['grit_media'])**2 \
                    + ((ego.std(ddof=1) - real['grit_sd']) / real['grit_sd'])**2 \
                    + ((r_gml - real['grit_ml']) / real['grit_ml'])**2
                if mejor_g is None or err < mejor_g[0]:
                    mejor_g = (err, w, centro, escala, ego.mean(), ego.std(ddof=1), r_gml)
    _, w_g, centro_g, escala_g, gm, gsd, r_gml = mejor_g
    grit = dict(w=round(float(w_g), 3), centro=round(float(centro_g), 3),
                escala=round(float(escala_g), 3))

    # ---- tabla de calibración (muestra seleccionada vs real)
    filas = list(fila_norm)
    for k, dim in enumerate(DIMS_BEPE):
        for met, rv, sv in [('media_seleccionada', medias_r[k], m_s[k]),
                            ('sd_seleccionada', sds_r[k], sd_s[k]),
                            ('sat_pc95', sat_real[k], sat_s[k])]:
            filas.append(dict(dim=dim, metrica=met, real=rv, sintetico=sv,
                              error_pct=100 * (sv - rv) / rv if rv else np.nan))
    for met, rv, sv in [('intercorr_media', real['intercorr'], ic_s),
                        ('grit_media', real['grit_media'], gm),
                        ('grit_sd', real['grit_sd'], gsd),
                        ('grit_r_ML', real['grit_ml'], r_gml)]:
        filas.append(dict(dim='GLOBAL', metrica=met, real=rv, sintetico=sv,
                          error_pct=100 * (sv - rv) / rv))
    tabla = pd.DataFrame(filas)
    tabla['pasa_10pct'] = tabla['error_pct'].abs() <= 10
    tabla.to_csv(OUT / 'cap7_calibracion_tabla_n41.csv', index=False, encoding='utf-8-sig')

    # ---- betas por (d, tau): d en población general (megamuestra N(0,1));
    # tasa base en la muestra analizada de cada panel:
    #   τ=1 → población general limpia (sin δ, sin selección);
    #   τ=0,5 y τ* → pool emprendedor (δ_g) + selección ruidosa (ρ*)
    g_mega = np.random.default_rng(s_mega).standard_normal(N_MEGA)
    suma_pool = pd_pool.sum(axis=1)
    z_sum_pool = (suma_pool - suma_pool.mean()) / suma_pool.std()
    v_pool = rho_star * z_sum_pool + np.sqrt(1 - rho_star**2) * eps
    orden_v = np.argsort(-v_pool)
    g_sel = {1.0: g_pop}
    for tau in [0.5, tau_star]:
        g_sel[tau] = g_pool[orden_v[:int(N_POOL_CAL * tau)]] if tau < 1 else g_pool
    betas = []
    for tau, gs_ in g_sel.items():
        for d_obj in D_GRID:
            b0, b1 = calibrar_celda(d_obj, gs_, g_mega)
            betas.append(dict(tau=round(tau, 4), d=d_obj, beta0=b0, beta1=b1,
                              d_verificado=d_general(b0, b1, g_mega),
                              tasa_seleccionada=float(enlace(b0 + b1 * gs_).mean())))
    tabla_b = pd.DataFrame(betas)
    tabla_b.to_csv(OUT / 'cap7_calibracion_betas_n41.csv', index=False, encoding='utf-8-sig')

    cfg = dict(tau_star=float(tau_star), rho_sel=float(rho_star),
               delta_g=float(delta_g), delta_s=delta_s.tolist(),
               cortes_pc95=cortes.tolist(),
               thr_dim=thr_dim.tolist(), grit=grit, p95=p95.tolist(),
               seed_raiz=SEED_RAIZ, n_pool_cal=N_POOL_CAL, n_mega=N_MEGA)
    (OUT / 'cap7_config_n41.json').write_text(json.dumps(cfg, indent=1))
    print(f'Cortes Pc95 (tabla normativa, límite inferior consistente): '
          f'{dict(zip(DIMS_BEPE, cortes))}')
    print(f'Percentil 95 interno del simulador: '
          f'{dict(zip(DIMS_BEPE, np.round(p95, 1)))}')
    print(f'δ_g = {delta_g} | δ_s por dimensión: '
          f'{dict(zip(DIMS_BEPE, np.round(delta_s, 2)))}')

    # ---- figura
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    xx = np.arange(8)
    for ax, (titulo, rv, sv) in zip(axes, [
            ('Medias PD (seleccionados)', medias_r, m_s),
            ('SD PD (seleccionados)', sds_r, sd_s),
            ('% Pc≥95 — saturación', sat_real, sat_s)]):
        ax.bar(xx - 0.2, rv, 0.4, label='real (n=41)', color='tab:blue')
        ax.bar(xx + 0.2, sv, 0.4, label='sintético (τ*, ρ, d=0)', color='tab:orange')
        ax.set_xticks(xx); ax.set_xticklabels(DIMS_BEPE, fontsize=8)
        ax.set_title(titulo, fontsize=10); ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Calibración Paso 4 — δ_g = {delta_g}, τ* = {tau_star}, ρ = {rho_star} | '
                 f'intercorr: {real["intercorr"]:.3f} vs {ic_s:.3f} | '
                 f'grit: {real["grit_media"]:.2f} vs {gm:.2f} | '
                 f'r grit-ML: {real["grit_ml"]:.2f} vs {r_gml:.2f}', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'cap7_calibracion_figura_n41.png', dpi=200)

    n_falla = int((~tabla['pasa_10pct']).sum())
    print(f'τ* = {tau_star}, ρ = {rho_star}  ({time.perf_counter()-t0:.0f} s)')
    print(tabla.round(3).to_string(index=False))
    print(f'\nMétricas fuera de ±10%: {n_falla} de {len(tabla)}')
    print(tabla_b.round(3).to_string(index=False))


# ---------------------------------------------------------------- piloto
def cargar_cfg():
    cfg = json.loads((OUT / 'cap7_config_n41.json').read_text())
    return (cfg, np.array(cfg['thr_dim']), cfg['grit'], cfg['rho_sel'],
            cfg['tau_star'], cfg['delta_g'], np.array(cfg['delta_s']))


def piloto(args):
    cfg, thr_dim, grit, rho, tau_star, delta_g, delta_s = cargar_cfg()
    tabla_b = pd.read_csv(OUT / 'cap7_calibracion_betas_n41.csv')
    fila = tabla_b[(tabla_b['d'] == 0.35) & (abs(tabla_b['tau'] - tau_star) < 1e-3)].iloc[0]
    b0, b1 = float(fila['beta0']), float(fila['beta1'])

    ss = np.random.SeedSequence(SEED_RAIZ + 7)
    res = []
    for outer, ns in [('kfold', [41, 120, 300, 500]), ('loo', [41, 80, 120])]:
        for n in ns:
            sems = ss.spawn(3)
            t0 = time.perf_counter()
            for sem in sems:
                replica_cap7(n, tau_star, 0.35, b0, b1, sem, outer, LAM_G, LAM_S,
                             RES_SD, DIM_IDX, thr_dim, grit, rho, delta_g,
                             delta_s, IDX_AM, MIN_MINORIA)
            t_med = (time.perf_counter() - t0) / 3
            res.append(dict(outer=outer, n=n, t_replica_s=round(t_med, 2)))
            print(f'  outer={outer:>5}  n={n:>3}  t/réplica = {t_med:6.2f} s', flush=True)

    tp = pd.DataFrame(res)
    presupuesto = {}
    for outer in ['kfold', 'loo']:
        sub = tp[tp['outer'] == outer]
        coef = np.polyfit(sub['n'], sub['t_replica_s'], 2)
        t_fn = lambda n: max(float(np.polyval(coef, n)), 0.05)
        total_s = sum(15 * M_REP * t_fn(n) for n in N_GRID)   # 15 celdas (5d×3τ) por n
        presupuesto[outer] = dict(
            horas_secuencial=round(total_s / 3600, 1),
            horas_16_nucleos=round(total_s / 3600 / 16, 1),
            t_por_n={str(n): round(t_fn(n), 2) for n in N_GRID})
    (OUT / 'cap7_piloto_n41.json').write_text(json.dumps(
        dict(mediciones=res, presupuesto_60000_replicas=presupuesto), indent=1))
    print(json.dumps(presupuesto, indent=1))


# ---------------------------------------------------------------- rejilla
def rejilla(args):
    cfg, thr_dim, grit, rho, tau_star, delta_g, delta_s = cargar_cfg()
    taus = [1.0, 0.5, round(tau_star, 4)]
    tabla_b = pd.read_csv(OUT / 'cap7_calibracion_betas_n41.csv')
    n_grid = [args.n_solo] if args.n_solo else [n for n in N_GRID if n <= args.n_max]
    archivo = OUT / ('cap7_replicas_loo41_n41.csv' if args.outer == 'loo'
                     else 'cap7_replicas_n41.csv')
    # semillas: la fila LOO usa su propia raíz para no reutilizar las de la rejilla
    raiz = SEED_RAIZ + (200 if args.outer == 'loo' else 100)
    ss = np.random.SeedSequence(raiz)
    celdas = [(n, tau, d) for tau in taus for d in D_GRID for n in n_grid]
    print(f'Rejilla: {len(celdas)} celdas × M={args.m}, outer={args.outer}, '
          f'semilla raíz={raiz}', flush=True)
    todas = []
    for i, (n, tau, d) in enumerate(celdas):
        fila = tabla_b[(tabla_b['d'] == d) & (abs(tabla_b['tau'] - tau) < 1e-3)].iloc[0]
        b0, b1 = float(fila['beta0']), float(fila['beta1'])
        sems = ss.spawn(args.m)
        t0 = time.perf_counter()
        # τ=1 = población limpia (sin perfil de pool ni selección)
        d_g = delta_g if tau < 1 else 0.0
        d_s = delta_s if tau < 1 else np.zeros(8)
        lote = Parallel(n_jobs=-1)(
            delayed(replica_cap7)(n, tau, d, b0, b1, sem, args.outer, LAM_G, LAM_S,
                                  RES_SD, DIM_IDX, thr_dim, grit, rho, d_g, d_s,
                                  IDX_AM, MIN_MINORIA)
            for sem in sems)
        todas.extend(lote)
        pd.DataFrame(todas).to_csv(archivo, index=False)  # checkpoint
        print(f'[{i+1:>3}/{len(celdas)}] n={n:>3} τ={tau:<6} d={d:<4} '
              f'({time.perf_counter()-t0:6.1f} s)', flush=True)
    print('Rejilla completa.')


# ---------------------------------------------------------------- productos
def wilson(k, m, z=1.96):
    p = k / m
    den = 1 + z**2 / m
    c = (p + z**2 / (2 * m)) / den
    w = z * np.sqrt(p * (1 - p) / m + z**2 / (4 * m**2)) / den
    return c - w, c + w


def productos(args):
    rep = pd.read_csv(OUT / 'cap7_replicas_n41.csv')
    tau_star = round(cargar_cfg()[4], 4)

    celdas = []
    for (n, tau), grupo in rep.groupby(['n', 'tau']):
        p95 = float(np.percentile(grupo[grupo['d'] == 0.0]['auc_cv'], 95))
        for d, gd in grupo.groupby('d'):
            k, m = int((gd['auc_cv'] > p95).sum()), len(gd)
            lo, hi = wilson(k, m)
            celdas.append(dict(n=n, tau=tau, d=d, M=m, p95_nulo=p95,
                               potencia=k / m, wilson_inf=lo, wilson_sup=hi,
                               auc_mediana=float(gd['auc_cv'].median()),
                               auc_q25=float(gd['auc_cv'].quantile(.25)),
                               auc_q75=float(gd['auc_cv'].quantile(.75)),
                               resub_gap_medio=float((gd['auc_resub'] - gd['auc_cv']).mean())))
    tc = pd.DataFrame(celdas)
    tc.to_csv(OUT / 'cap7_celdas_n41.csv', index=False, encoding='utf-8-sig')

    real = tc[(tc['n'] == 41) & (abs(tc['tau'] - tau_star) < 1e-3)]
    print('=== Potencia del caso real (n=41, τ*) ===')
    print(real.round(3).to_string(index=False))
    real.to_csv(OUT / 'cap7_potencia_caso_real_n41.csv', index=False, encoding='utf-8-sig')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = tc[abs(tc['tau'] - tau_star) < 1e-3]
    for d, gd in sub.groupby('d'):
        gd = gd.sort_values('n')
        ax.plot(gd['n'], gd['auc_mediana'], marker='o', label=f'd = {d}')
        ax.fill_between(gd['n'], gd['auc_q25'], gd['auc_q75'], alpha=0.15)
    ax.axvline(41, color='seagreen', ls=':', lw=1.5)
    g35 = sub[sub['d'] == 0.35].sort_values('n')
    n80 = float(np.interp(0.80, g35['potencia'], g35['n'],
                          left=np.nan, right=np.nan))
    eti = f'≈{n80:.0f}' if np.isfinite(n80) else f'> {int(g35["n"].max())}'
    ax.set_xscale('log'); ax.set_xticks(sorted(sub['n'].unique()))
    ax.get_xaxis().set_major_formatter('{x:.0f}')
    ax.set_xlabel('n (escala log)'); ax.set_ylabel('AUC_cv (mediana, banda IQR)')
    ax.set_title(f'Curva de aprendizaje (τ* = {tau_star}); '
                 f'n para potencia 0.80 con d = 0.35: {eti}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'cap7_curva_aprendizaje_n41.png', dpi=200)

    taus = sorted(tc['tau'].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(taus), figsize=(5 * len(taus), 4.4), sharey=True)
    for ax, tau in zip(np.atleast_1d(axes), taus):
        piv = tc[abs(tc['tau'] - tau) < 1e-9].pivot(index='d', columns='n',
                                                    values='potencia')
        im = ax.imshow(piv.to_numpy(), origin='lower', cmap='viridis',
                       vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=8)
        for ii in range(piv.shape[0]):
            for jj in range(piv.shape[1]):
                ax.text(jj, ii, f'{piv.iloc[ii, jj]:.2f}', ha='center',
                        va='center', fontsize=7, color='white')
        es_real = abs(tau - tau_star) < 1e-6
        ax.set_title('τ* (caso real)' if es_real else f'τ = {tau}', fontsize=10)
        ax.set_xlabel('n')
        if es_real and 41 in list(piv.columns):
            jj = list(piv.columns).index(41)
            ax.add_patch(plt.Rectangle((jj - .5, -.5), 1, piv.shape[0],
                                       fill=False, edgecolor='red', lw=2))
    np.atleast_1d(axes)[0].set_ylabel('d verdadero')
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label='potencia', shrink=0.85)
    fig.suptitle(f'Mapa de detectabilidad (nula empírica, M={int(tc["M"].median())})')
    fig.savefig(OUT / 'cap7_mapa_detectabilidad_n41.png', dpi=200)
    print(f'n para potencia 0.80 (d=0.35, τ*): {eti}')

    # ---- validación cruzada Thorndike (Cap. 6 vs simulador): correlación
    # PD_k–g verdadera (pool completo) vs atenuada (seleccionados) vs
    # corregida con Thorndike caso II usando U = SD_pool/SD_sel
    cfg, thr_dim, grit, rho, tau_star_f, delta_g, delta_s = cargar_cfg()
    s_cal, _, s_grit = np.random.SeedSequence(SEED_RAIZ).spawn(3)
    g_pop, s_pop, lat_pop = latentes(N_POOL_CAL, np.random.default_rng(s_cal))
    lat_pool = lat_pop + delta_g * LAM_G[None, :] + delta_s[DIM_IDX] * LAM_S[None, :]
    pd_pool = discretizar(lat_pool, thr_dim)
    g_pool = g_pop + delta_g
    eps = np.random.default_rng(s_grit).standard_normal(N_POOL_CAL)
    suma = pd_pool.sum(1)
    v = rho * (suma - suma.mean()) / suma.std() + np.sqrt(1 - rho**2) * eps
    orden_v = np.argsort(-v)
    filas_t = []
    for tau in [0.5, round(tau_star_f, 4)]:
        sel = orden_v[:int(N_POOL_CAL * tau)]
        for k, dim in enumerate(DIMS_BEPE):
            r_pop = float(np.corrcoef(pd_pool[:, k], g_pool)[0, 1])
            r_sel = float(np.corrcoef(pd_pool[sel, k], g_pool[sel])[0, 1])
            u = pd_pool[:, k].std(ddof=1) / pd_pool[sel, k].std(ddof=1)
            r_corr = r_sel * u / np.sqrt(1 + r_sel**2 * (u**2 - 1))
            filas_t.append(dict(tau=tau, dim=dim, r_poblacional=r_pop,
                                r_seleccionada=r_sel, U=u,
                                r_corregida_thorndike=r_corr,
                                error_correccion=r_corr - r_pop))
    tt = pd.DataFrame(filas_t)
    tt.to_csv(OUT / 'cap7_thorndike_n41.csv', index=False, encoding='utf-8-sig')
    print('\n=== Validación cruzada Thorndike (simulador vs corrección Cap. 6) ===')
    print(tt.round(3).to_string(index=False))

    fig, axes2 = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tau in zip(axes2, [0.5, round(tau_star_f, 4)]):
        sub2 = tt[tt['tau'] == tau]
        xx2 = np.arange(8)
        ax.bar(xx2 - 0.25, sub2['r_poblacional'], 0.25, label='r verdadera (pool)',
               color='tab:green')
        ax.bar(xx2, sub2['r_seleccionada'], 0.25, label='r atenuada (seleccionados)',
               color='tab:red')
        ax.bar(xx2 + 0.25, sub2['r_corregida_thorndike'], 0.25,
               label='r corregida (Thorndike II)', color='tab:blue')
        ax.set_xticks(xx2); ax.set_xticklabels(DIMS_BEPE, fontsize=8)
        ax.set_title(f'τ = {tau}', fontsize=10); ax.grid(axis='y', alpha=0.3)
    axes2[0].set_ylabel('correlación PD–g'); axes2[0].legend(fontsize=8)
    fig.suptitle('Validación cruzada: atenuación por truncamiento, simulador vs fórmula')
    fig.tight_layout()
    fig.savefig(OUT / 'cap7_thorndike_n41.png', dpi=200)


# ---------------------------------------------------------------- validación 8.1
def validar(args):
    rep = pd.read_csv(OUT / 'cap7_replicas_n41.csv')
    print('=== Paso 8.1 — validación del simulador ===')
    ok = True

    # banda correcta del estadístico split-half bajo H0: incluye el ruido de
    # estimar el p95 con media muestra (Wilson con umbral fijo es demasiado
    # estrecho y sobre-señala)
    rng = np.random.default_rng(1)
    sim = rng.random((100_000, 500))
    det0 = (sim[:, 1::2] > np.percentile(sim[:, ::2], 95, axis=1)[:, None]).mean(1)
    lo_b, hi_b = np.percentile(det0, [2.5, 97.5])
    print(f'  banda split-half bajo H0 (2.5–97.5%): [{lo_b:.3f}, {hi_b:.3f}]')

    fuera = 0
    for (n, tau), grupo in rep.groupby(['n', 'tau']):
        nulo = grupo[grupo['d'] == 0.0]['auc_cv'].to_numpy()
        det_mc = float((nulo > np.percentile(nulo, 95)).mean())   # preregistrado
        det_sh = float((nulo[1::2] > np.percentile(nulo[::2], 95)).mean())
        marca = 'OK' if lo_b <= det_sh <= hi_b else 'fuera'
        fuera += marca != 'OK'
        print(f'  d=0  n={n:>3} τ={tau:<6}: mismo-celda = {det_mc:.3f} | '
              f'split-half = {det_sh:.3f} {marca}')
    # bajo H0 se esperan ~1.2 de 24 fuera; se exige no más de 3 (p≈0.08)
    print(f'  celdas fuera de banda: {fuera} de 24 (esperado ≈1; límite 3)')
    ok &= fuera <= 3
    extra = OUT / 'cap7_nulos_extra_n41.csv'
    if extra.exists():
        nx = pd.read_csv(extra)
        for (n, tau), g in nx.groupby(['n', 'tau']):
            orig = rep[(rep['n'] == n) & (rep['tau'] == tau) &
                       (rep['d'] == 0.0)]['auc_cv']
            det = float((g['auc_cv'] > np.percentile(orig, 95)).mean())
            print(f'  refuerzo nulas frescas n={n} τ={tau}: detección = {det:.3f}')
    cel = rep[(rep['n'] == 500) & (rep['tau'] == 1.0) & (rep['d'] == 0.8)]
    if len(cel):
        nulo = rep[(rep['n'] == 500) & (rep['tau'] == 1.0) & (rep['d'] == 0.0)]['auc_cv']
        pot = float((cel['auc_cv'] > np.percentile(nulo, 95)).mean())
        med = float(cel['auc_cv'].median())
        print(f'  Recuperación (n=500, d=0.8, τ=1): potencia={pot:.3f} '
              f'(esperado ≈1), AUC mediana={med:.3f} (teórico 0.714)')
        ok &= pot >= 0.95 and abs(med - 0.714) <= 0.05
    print('VALIDACIÓN', 'SUPERADA' if ok else 'FALLIDA — revisar antes de reportar')


# ---------------------------------------------------------------- sensibilidad
SENS_N = [41, 80, 120, 200, 300, 500]
SENS_D = [0.0, 0.35, 0.8]


def sensibilidad(args):
    """Paso 8.2: variantes preregistradas, rejilla reducida en el panel τ*.
    Cada variante usa su propia nula empírica (d=0) por n."""
    cfg, thr_dim, grit, rho, tau_star, delta_g, delta_s = cargar_cfg()
    tabla_b = pd.read_csv(OUT / 'cap7_calibracion_betas_n41.csv')

    # pool determinista de calibración (mismas semillas que --calibrar)
    s_cal, s_mega, s_grit = np.random.SeedSequence(SEED_RAIZ).spawn(3)
    g_pop, s_pop, lat_pop = latentes(N_POOL_CAL, np.random.default_rng(s_cal))
    lat_pool = lat_pop + delta_g * LAM_G[None, :] + delta_s[DIM_IDX] * LAM_S[None, :]
    pd_pool = discretizar(lat_pool, thr_dim)
    eps = np.random.default_rng(s_grit).standard_normal(N_POOL_CAL)
    suma = pd_pool.sum(1)
    v = rho * (suma - suma.mean()) / suma.std() + np.sqrt(1 - rho**2) * eps
    sel = np.argsort(-v)[:int(N_POOL_CAL * tau_star)]
    g_sel = (g_pop + delta_g)[sel]
    f_esp_sel = ((s_pop + delta_s)[sel][:, IDX_ST] +
                 (s_pop + delta_s)[sel][:, IDX_AM]) / np.sqrt(2)
    g_mega = np.random.default_rng(s_mega).standard_normal(N_MEGA)

    # cópula gaussiana de la muestra real (sin selección: ya es población
    # seleccionada); efecto en el primer componente latente normalizado
    df = pd.read_excel(DATA).dropna(subset=['exito'])
    cols9 = [f'BEPE_{k}_PD' for k in DIMS_BEPE] + ['ego']
    x_real = df[cols9].to_numpy(float)
    r_cop = np.corrcoef(x_real.T) + np.eye(9) * 1e-9
    chol = np.linalg.cholesky(r_cop)
    margenes = np.sort(x_real.T, axis=1)
    vals, vecs = np.linalg.eigh(r_cop)
    w1 = vecs[:, -1] / np.sqrt(vals[-1])      # Var(Z·w1) = 1
    if (vecs[:, -1].sum()) < 0:
        w1 = -w1

    # medias por enseña para la variante de mezcla (conversión PD→latente
    # aproximada dPD/dδ ≈ 1,3; registrada como aproximación)
    medias_tot = df[[f'BEPE_{k}_PD' for k in DIMS_BEPE]].mean().to_numpy()
    delta_ens, pesos_ens = {}, {}
    for ens, sub in df.groupby('franquicia'):
        m_e = sub[[f'BEPE_{k}_PD' for k in DIMS_BEPE]].mean().to_numpy()
        delta_ens[ens] = delta_s + (m_e - medias_tot) / 1.3
        pesos_ens[ens] = len(sub) / len(df)
    ens_a, ens_b = sorted(pesos_ens, key=pesos_ens.get, reverse=True)

    def betas_variante(variante):
        out = {}
        for d in SENS_D:
            if variante == 'copula':
                out[d] = calibrar_celda(d, g_mega, g_mega)
            elif variante == 'especificos':
                out[d] = calibrar_celda(d, f_esp_sel, g_mega)
            elif variante == 'tasa50':
                out[d] = calibrar_celda(d, g_sel, g_mega, tasa=0.5)
            elif variante == 'probit':
                out[d] = calibrar_celda(d, g_sel, g_mega, link='probit')
            else:                              # mezcla, M1000: betas del principal
                fila = tabla_b[(tabla_b['d'] == d) &
                               (abs(tabla_b['tau'] - tau_star) < 1e-3)].iloc[0]
                out[d] = (float(fila['beta0']), float(fila['beta1']))
        return out

    variantes = ['copula', 'especificos', 'tasa50', 'probit', 'mezcla', 'M1000']
    ss = np.random.SeedSequence(SEED_RAIZ + 300)
    todas = []
    registro_betas = []
    for vi, variante in enumerate(variantes):
        betas = betas_variante(variante)
        for d, (b0, b1) in betas.items():
            registro_betas.append(dict(variante=variante, d=d, beta0=b0, beta1=b1))
        ns = [41] if variante == 'M1000' else SENS_N
        m = 1000 if variante == 'M1000' else args.m
        for n in ns:
            for d in SENS_D:
                b0, b1 = betas[d]
                sems = ss.spawn(m)
                t0 = time.perf_counter()
                if variante == 'copula':
                    lote = Parallel(n_jobs=-1)(
                        delayed(replica_copula)(n, d, b0, b1, sem, chol, margenes,
                                                w1, MIN_MINORIA) for sem in sems)
                else:
                    kw = {}
                    if variante == 'especificos':
                        kw = dict(efecto='especificos')
                    elif variante == 'probit':
                        kw = dict(link='probit')
                    elif variante == 'mezcla':
                        kw = dict(delta_s2=delta_ens[ens_b],
                                  peso_mezcla=pesos_ens[ens_a])
                    d_s = delta_ens[ens_a] if variante == 'mezcla' else delta_s
                    lote = Parallel(n_jobs=-1)(
                        delayed(replica_cap7)(n, tau_star, d, b0, b1, sem, 'kfold',
                                              LAM_G, LAM_S, RES_SD, DIM_IDX, thr_dim,
                                              grit, rho, delta_g, d_s, IDX_AM,
                                              MIN_MINORIA, **kw) for sem in sems)
                for fila in lote:
                    fila['variante'] = variante
                todas.extend(lote)
                pd.DataFrame(todas).to_csv(OUT / 'cap7_sens_replicas_n41.csv',
                                           index=False)
                print(f'[{variante:>12}] n={n:>3} d={d:<4} M={m} '
                      f'({time.perf_counter()-t0:6.1f} s)', flush=True)

    pd.DataFrame(registro_betas).to_csv(OUT / 'cap7_sens_betas_n41.csv',
                                        index=False, encoding='utf-8-sig')

    # resumen: potencia contra la nula empírica de cada variante y n
    rep = pd.DataFrame(todas)
    filas = []
    for (variante, n), grupo in rep.groupby(['variante', 'n']):
        p95 = float(np.percentile(grupo[grupo['d'] == 0.0]['auc_cv'], 95))
        for d, gd in grupo.groupby('d'):
            k, m = int((gd['auc_cv'] > p95).sum()), len(gd)
            lo, hi = wilson(k, m)
            filas.append(dict(variante=variante, n=n, d=d, M=m, p95_nulo=p95,
                              potencia=k / m, wilson_inf=lo, wilson_sup=hi,
                              auc_mediana=float(gd['auc_cv'].median())))
    resumen = pd.DataFrame(filas)
    resumen.to_csv(OUT / 'cap7_sensibilidad_resumen_n41.csv', index=False,
                   encoding='utf-8-sig')
    print(resumen.round(3).to_string(index=False))

    # figura: curvas de potencia para d=0.35, variante vs principal
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    try:
        prin = pd.read_csv(OUT / 'cap7_celdas_n41.csv')
        prin = prin[(abs(prin['tau'] - round(tau_star, 4)) < 1e-3) &
                    (prin['d'] == 0.35)].sort_values('n')
        ax.plot(prin['n'], prin['potencia'], 'k-o', lw=2.2,
                label='principal (bifactor, τ*)')
    except FileNotFoundError:
        pass
    for variante, gd in resumen[resumen['d'] == 0.35].groupby('variante'):
        if variante == 'M1000':
            continue
        gd = gd.sort_values('n')
        ax.plot(gd['n'], gd['potencia'], marker='s', alpha=0.8, label=variante)
    ax.axhline(0.8, color='firebrick', ls='--', lw=1)
    ax.axvline(41, color='seagreen', ls=':', lw=1.5)
    ax.set_xscale('log'); ax.set_xticks(SENS_N)
    ax.get_xaxis().set_major_formatter('{x:.0f}')
    ax.set_xlabel('n (escala log)'); ax.set_ylabel('potencia (d = 0.35)')
    ax.set_title('Sensibilidad 8.2: potencia para d = 0.35 según variante')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'cap7_sensibilidad_figura_n41.png', dpi=200)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calibrar', action='store_true')
    ap.add_argument('--piloto', action='store_true')
    ap.add_argument('--rejilla', action='store_true')
    ap.add_argument('--productos', action='store_true')
    ap.add_argument('--validar', action='store_true')
    ap.add_argument('--sensibilidad', action='store_true')
    ap.add_argument('--outer', choices=['kfold', 'loo'], default='kfold')
    ap.add_argument('--m', type=int, default=M_REP)
    ap.add_argument('--n-max', type=int, default=500)
    ap.add_argument('--n-solo', type=int, default=None,
                    help='restringe la rejilla a un único n (fila LOO de robustez)')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.calibrar:
        calibrar(args)
    if args.piloto:
        piloto(args)
    if args.rejilla:
        rejilla(args)
    if args.productos:
        productos(args)
    if args.validar:
        validar(args)
    if args.sensibilidad:
        sensibilidad(args)


if __name__ == '__main__':
    main()
