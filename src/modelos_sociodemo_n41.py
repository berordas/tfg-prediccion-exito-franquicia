# -*- coding: utf-8 -*-
"""Dos modelos adicionales del Capítulo 5 (mismo pipeline ridge de referencia:
StandardScaler + LogisticRegression(C=1), outer LOO, AUC resustitución y
validado): (a) solo sociodemográficas; (b) sociodemográficas + 9 psicométricas.
Sociodemográficas categóricas en one-hot (drop_first); edad numérica."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / 'data' / 'dataset_franquicias_seudonimizado.xlsx'

d = pd.read_excel(DATA).dropna(subset=['exito'])
y = d['exito'].astype(int).to_numpy()

SOCIO_CAT = ['genero', 'nivel_educativo', 'años_experiencia', 'lidero_equipos',
             'estado_civil', 'hijos_dependientes']
PSICO = [f'BEPE_{x}_PD' for x in ['AE', 'AU', 'IN', 'LI', 'ML', 'OP', 'TE', 'TR']] + ['ego']

socio = pd.get_dummies(d[SOCIO_CAT], drop_first=True).astype(float)
socio.insert(0, 'edad', d['edad'].astype(float).to_numpy())

X_a = socio.to_numpy()
X_b = np.column_stack([socio.to_numpy(), d[PSICO].to_numpy(float)])

pipe = Pipeline([('sc', StandardScaler()),
                 ('lr', LogisticRegression(C=1.0, max_iter=2000))])

for nombre, X in [('(a) sociodemográficas', X_a),
                  ('(b) combinado socio + psicométricas', X_b)]:
    proba = cross_val_predict(pipe, X, y, cv=LeaveOneOut(),
                              method='predict_proba')[:, 1]
    auc_loo = roc_auc_score(y, proba)
    pipe.fit(X, y)
    auc_resub = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    print(f'{nombre}: {X.shape[1]} predictores | '
          f'AUC resustitución = {auc_resub:.3f} | AUC LOO = {auc_loo:.3f}')
