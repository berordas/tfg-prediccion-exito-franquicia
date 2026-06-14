# -*- coding: utf-8 -*-
"""Matriz de intercorrelaciones (Pearson) de las 8 dimensiones BEPE en
puntuación directa, sobre la muestra etiquetada (n=41). Citada en §5.1 y
en el Apéndice B de la memoria.

Salidas (resultados/):
  - intercorrelaciones_bepe_n41.csv : matriz 8×8 redondeada a 2 decimales
  - intercorrelaciones_bepe_n41.png : heatmap con los valores anotados
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dataset_franquicias_seudonimizado.xlsx'
OUT = ROOT / 'resultados'
OUT.mkdir(exist_ok=True)

COLS = ['BEPE_AE_PD', 'BEPE_AU_PD', 'BEPE_IN_PD', 'BEPE_LI_PD',
        'BEPE_ML_PD', 'BEPE_OP_PD', 'BEPE_TE_PD', 'BEPE_TR_PD']
LABELS = ['AE', 'AU', 'IN', 'LI', 'ML', 'OP', 'TE', 'TR']

df = pd.read_excel(DATA)
df = df[df['exito'].notna()]                 # solo filas etiquetadas (n=41)

M = df[COLS].corr().round(2)
M.index = LABELS
M.columns = LABELS
M.to_csv(OUT / 'intercorrelaciones_bepe_n41.csv')

m = M.to_numpy()
off_diag = m[~np.eye(len(m), dtype=bool)].mean()
print(f'n = {len(df)}  |  intercorrelación media (fuera de la diagonal) = {off_diag:.3f}')
print(M.to_string())

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(m, vmin=-1, vmax=1, cmap='coolwarm')
ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center', fontsize=8)
ax.set_title('Matriz de intercorrelaciones de las dimensiones BEPE (n=41)')
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUT / 'intercorrelaciones_bepe_n41.png', dpi=150)
print(f'Guardado en: {OUT}')
