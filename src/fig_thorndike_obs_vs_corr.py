# -*- coding: utf-8 -*-
"""Figura 3 de la memoria (§6, Tabla 4): correlaciones de las 8 dimensiones
BEPE con el criterio, observadas frente a corregidas por la fórmula de
Thorndike caso II. Lee los valores reales del JSON (cap6_thorndike), de modo
que coincide exactamente con la Tabla 4.

NO confundir con cap7_thorndike_n41.png, que es la validación cruzada del
simulador (atenuación por truncamiento: simulador vs fórmula).

Salida: resultados/fig_thorndike_obs_vs_corr_n41.png
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / 'resultados' / 'resultados_cap456_n41.json', encoding='utf-8'))
t = d['cap6_thorndike']
dims = [x['dim'] for x in t]
r_obs = [x['r_obs'] for x in t]
r_cor = [x['r_corregida'] for x in t]

x = np.arange(len(dims)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w / 2, r_obs, w, label='r observada')
ax.bar(x + w / 2, r_cor, w, label='r corregida (Thorndike II)')
ax.axhline(0, color='grey', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(dims)
ax.set_ylabel('correlación con el criterio')
ax.set_title('Corrección de Thorndike: correlaciones observadas frente a corregidas')
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / 'resultados' / 'fig_thorndike_obs_vs_corr_n41.png', dpi=150)
print(f'n dims = {len(dims)}')
for dim, ro, rc in zip(dims, r_obs, r_cor):
    print(f'  {dim}: r_obs={ro:+.3f}  r_corregida={rc:+.3f}')
print(f'Guardado en: {ROOT / "resultados" / "fig_thorndike_obs_vs_corr_n41.png"}')
