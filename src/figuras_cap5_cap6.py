# -*- coding: utf-8 -*-
"""Dos figuras para la memoria, con el estilo de las del Capítulo 7:
(a) §5.2 — saturación de percentiles de las dimensiones BEPE frente a lo
    esperado en la población normativa (5% sobre Pc95, 1% en Pc99);
(b) §6.3 — distribución nula del AUC del test de permutación (1.000
    permutaciones, ridge+LOO) con el AUC observado marcado."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data' / 'dataset_franquicias_seudonimizado.xlsx'
OUT = BASE / 'resultados'

DIMS = ['AE', 'AU', 'IN', 'LI', 'ML', 'OP', 'TE', 'TR', 'EMP']

d = pd.read_excel(DATA).dropna(subset=['exito'])

# ---------------------------------------------------------------- (a) saturación
p95 = [float((d[f'BEPE_{x}_Pc'] >= 95).mean() * 100) for x in DIMS]
p99 = [float((d[f'BEPE_{x}_Pc'] == 99).mean() * 100) for x in DIMS]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
xx = np.arange(len(DIMS))
ax.bar(xx - 0.2, p95, 0.4, label='muestra: % con Pc ≥ 95', color='tab:blue')
ax.bar(xx + 0.2, p99, 0.4, label='muestra: % con Pc = 99', color='tab:orange')
ax.axhline(5, color='firebrick', linestyle='--', linewidth=1,
           label='esperado en población normativa (5% sobre Pc95)')
ax.axhline(1, color='firebrick', linestyle=':', linewidth=1,
           label='esperado (1% en Pc99)')
ax.set_xticks(xx); ax.set_xticklabels(DIMS, fontsize=9)
ax.set_ylabel('% de la muestra (n = 41)')
ax.set_title('Saturación de percentiles BEPE frente a las normas '
             '(efecto techo, §5.2)')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig_saturacion_percentiles_n41.png', dpi=200)
print('(a) saturación:', dict(zip(DIMS, np.round(p95, 1))))

# ---------------------------------------------------------------- (b) permutación
nulos = np.load(OUT / 'null_aucs_real_n41.npy')
res = json.loads((OUT / 'resultados_cap456_n41.json').read_text(encoding='utf-8'))
auc_obs = res['cap5_modelos']['ridge']['auc_loo']
p_perm = res['cap5_modelos']['permutacion']['p']
p95_nulo = np.percentile(nulos, 95)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.hist(nulos, bins=40, color='lightsteelblue', edgecolor='steelblue',
        label=f'AUC bajo etiquetas permutadas (B = {len(nulos)})')
ax.axvline(auc_obs, color='firebrick', linewidth=2,
           label=f'AUC observado = {auc_obs:.3f} (p = {p_perm:.3f})')
ax.axvline(p95_nulo, color='dimgray', linestyle='--', linewidth=1.2,
           label=f'percentil 95 de la nula = {p95_nulo:.3f}')
ax.axvline(np.mean(nulos), color='navy', linestyle=':', linewidth=1.2,
           label=f'media de la nula = {np.mean(nulos):.3f}')
ax.set_xlabel('AUC validado (ridge + LOO)')
ax.set_ylabel('frecuencia')
ax.set_title('Test de permutación del pipeline ridge+LOO '
             f'(n = 41, 1.000 permutaciones, §6.3)')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig_permutacion_auc_n41.png', dpi=200)
print(f'(b) permutación: AUC obs = {auc_obs}, p = {p_perm}, '
      f'nula media = {np.mean(nulos):.3f}, p95 = {p95_nulo:.3f}')
