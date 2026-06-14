# -*- coding: utf-8 -*-
"""Refuerzo de validación 8.1: 500 réplicas nulas frescas (semillas raíz
20260611+400) para las celdas señaladas por el split-half, evaluadas contra
el p95 de las 500 réplicas originales (muestra totalmente independiente)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sim_capitulo7 import (replica_cap7, cargar_cfg, LAM_G, LAM_S, RES_SD,
                           DIM_IDX, IDX_AM, MIN_MINORIA, SEED_RAIZ, OUT, wilson)

CELDAS = [(41, 1.0), (120, 1.0)]   # las dos peores del split-half

if __name__ == '__main__':
    cfg, thr_dim, grit, rho, tau_star, delta_g, delta_s = cargar_cfg()
    betas = pd.read_csv(OUT / 'cap7_calibracion_betas_n41.csv')
    rep = pd.read_csv(OUT / 'cap7_replicas_n41.csv')
    ss = np.random.SeedSequence(SEED_RAIZ + 400)

    nuevas = []
    for n, tau in CELDAS:
        fila = betas[(betas['d'] == 0.0) & (abs(betas['tau'] - tau) < 1e-3)].iloc[0]
        b0, b1 = float(fila['beta0']), float(fila['beta1'])
        d_g = delta_g if tau < 1 else 0.0
        d_s = delta_s if tau < 1 else np.zeros(8)
        sems = ss.spawn(500)
        lote = Parallel(n_jobs=-1)(
            delayed(replica_cap7)(n, tau, 0.0, b0, b1, sem, 'kfold', LAM_G, LAM_S,
                                  RES_SD, DIM_IDX, thr_dim, grit, rho, d_g, d_s,
                                  IDX_AM, MIN_MINORIA)
            for sem in sems)
        for f in lote:
            f['lote'] = 'extra'
        nuevas.extend(lote)

        orig = rep[(rep['n'] == n) & (rep['tau'] == tau) & (rep['d'] == 0.0)]['auc_cv']
        p95 = np.percentile(orig, 95)
        det = float((pd.DataFrame(lote)['auc_cv'] > p95).mean())
        lo, hi = wilson(int(det * 500), 500)
        print(f'n={n} τ={tau}: detección de 500 nulas FRESCAS contra p95 original '
              f'= {det:.3f} (Wilson [{lo:.3f}, {hi:.3f}]; banda correcta '
              f'umbral+binomial ≈ [0.027, 0.082])')

    pd.DataFrame(nuevas).to_csv(OUT / 'cap7_nulos_extra_n41.csv', index=False)
