"""Figura: curva de potencia frente a n por tamaño del efecto (τ* = 0.9).

Genera resultados/cap7_curva_potencia_n41.png a partir de
resultados/cap7_celdas_n41.csv (mismo estilo que la curva de aprendizaje).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'resultados'
TAU_STAR = 0.9

tc = pd.read_csv(OUT / 'cap7_celdas_n41.csv')
sub = tc[abs(tc['tau'] - TAU_STAR) < 1e-3]

fig, ax = plt.subplots(figsize=(8, 5))
for d, gd in sub.groupby('d'):
    gd = gd.sort_values('n')
    ax.plot(gd['n'], gd['potencia'], marker='o', label=f'd = {d}')
    ax.fill_between(gd['n'], gd['wilson_inf'], gd['wilson_sup'], alpha=0.15)

ax.axhline(0.80, color='firebrick', ls='--', lw=1.2)
ax.text(320, 0.815, 'potencia objetivo = 0.80', color='firebrick', fontsize=8)
ax.axvline(41, color='seagreen', ls=':', lw=1.5)
ax.text(42, 0.62, 'caso real (n = 41)', color='seagreen', fontsize=8,
        rotation=90, va='center')
ax.annotate('MDE con n = 41: d ≈ 0.93 (AUC 0.745)', xy=(41, 0.80),
            xytext=(95, 0.63), fontsize=8,
            arrowprops=dict(arrowstyle='->', lw=0.8))

g35 = sub[sub['d'] == 0.35].sort_values('n')
n80 = float(np.interp(0.80, g35['potencia'], g35['n'], left=np.nan, right=np.nan))
eti = f'≈{n80:.0f}' if np.isfinite(n80) else f'> {int(g35["n"].max())}'
ax.set_xscale('log'); ax.set_xticks(sorted(sub['n'].unique()))
ax.get_xaxis().set_major_formatter('{x:.0f}')
ax.set_ylim(0, 1.05)
ax.set_xlabel('n (escala log)')
ax.set_ylabel('potencia (proporción de detecciones, banda de Wilson)')
ax.set_title(f'Curva de potencia (τ* = {TAU_STAR}); '
             f'n para potencia 0.80 con d = 0.35: {eti}')
ax.legend(fontsize=8, loc='center right'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / 'cap7_curva_potencia_n41.png', dpi=200)
print('OK ->', OUT / 'cap7_curva_potencia_n41.png')
