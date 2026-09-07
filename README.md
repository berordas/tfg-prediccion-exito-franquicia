# TFG — Viabilidad de la predicción del éxito emprendedor en franquicia (BEPE + GRIT)

Apéndice computacional del Trabajo de Fin de Grado *"Modelo de predicción del
éxito emprendedor en franquicia a partir de perfiles psicométricos e IA"*
(Bernardo Ordás Cernadas).

El trabajo pregunta si la personalidad emprendedora medida con la batería BEPE
(8 dimensiones) y el grit (EGO) permite predecir el éxito de un franquiciado, y
responde con dos análisis: (1) el análisis del dataset real (n = 41) y (2) un
estudio de potencia por simulación Monte Carlo que cuantifica con qué tamaño
muestral *sería* detectable un efecto realista. La conclusión es que, con la
muestra disponible, el procedimiento no puede detectar un efecto de tamaño
realista, y el estudio entrega el mapa de cuándo podría.

## Memoria

El documento del TFG —planteamiento, marco teórico, método, resultados y
discusión— está incluido en el repositorio:

📄 **[TFG — Ordás Cernadas, Bernardo (PDF)](resultados/TFG%20-%20Ord%C3%A1s%20Cernadas%2C%20Bernardo.pdf)**

> **Versión pública.** Las enseñas colaboradoras facilitaron sus criterios
> internos de éxito de forma confidencial, por lo que esta versión los sustituye
> por una formulación genérica en los apartados 4.4 y 6.4. El resto del documento
> se publica íntegro.

Lo que sigue es el **apéndice computacional** de esa memoria: el código que
genera las tablas y figuras que allí se citan. La correspondencia entre cada
salida y su sección está en [Mapa salidas → memoria](#mapa-salidas--memoria).

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/                          # NO incluido en el repo (datos privados) — ver «Datos»
├── src/
│   ├── analisis_tfg.py            # §5.1–5.3: correlaciones, modelos LOO, Thorndike, potencia, permutación
│   ├── figuras_cap5_cap6.py       # Figuras §5.1.2 (saturación) y §5.2.3 (nula de permutación)
│   ├── modelos_sociodemo_n41.py   # §5.2: modelos logísticos sociodemográfico y combinado (AUC resub/LOO)
│   ├── sim_capitulo7.py           # §5.4: simulación Monte Carlo completa (calibración → rejilla → productos → sensibilidad)
│   ├── cap7_nulos_extra.py        # §5.4: refuerzo de validación (nulas frescas independientes)
│   └── sim_tfg.py                 # Procedencia: demo reducida de la dirección (no forma parte de la reproducción)
└── resultados/
    ├── TFG - Ordás Cernadas, Bernardo.pdf   # ← memoria (ver aviso arriba)
    └── ...                         # Salidas precalculadas (tablas, JSON, figuras) — se regeneran con los scripts
```

## Requisitos

- **Python 3.13** (desarrollado en 3.13.1).
- Dependencias en `requirements.txt`. Instalación:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

> `scikit-learn >= 1.8` es obligatorio (uso de `LogisticRegression(l1_ratio=...)`).

## Datos

**Los datos no se publican en este repositorio.** Por protección de datos no se
distribuyen aquí ni el dataset seudonimizado ni los datos crudos (informes BEPE
individuales, dataset original con correos, tabla de correspondencia); la carpeta
`data/` está ignorada por completo en `.gitignore`.

El fichero que consumen los scripts es `data/dataset_franquicias_seudonimizado.xlsx`:
45 filas (41 con la variable `exito` etiquetada: 26 éxito / 15 no-éxito; tasa base
0,634), IDs seudonimizados `S001`–`S045`, sin correos ni códigos reidentificables.
Para reejecutar los análisis hay que colocarlo en esa ruta. Sin él, los resultados
siguen siendo consultables: las tablas y figuras ya vienen precalculadas en
`resultados/` y comentadas en la [memoria](resultados/TFG%20-%20Ord%C3%A1s%20Cernadas%2C%20Bernardo.pdf).

Todos los scripts resuelven sus rutas con `Path(__file__)`, de modo que se
ejecutan desde cualquier directorio sin variables de entorno.

## Orden de ejecución

Las salidas ya vienen precalculadas en `resultados/`. Para regenerarlas desde
cero hace falta el dataset en `data/` (ver [Datos](#datos)) y ejecutar **desde la
raíz del repositorio** en este orden. Los tiempos son orientativos sobre 16
núcleos (la simulación paraleliza con `joblib`).

### Análisis del dataset real (§5.1–5.3)

```bash
# 1) Correlaciones, d de Cohen, modelos con LOO, Thorndike, potencia analítica
#    y test de permutación (1.000 réplicas, ridge+LOO).
python src/analisis_tfg.py            # ~2–5 min
#    → resultados/resultados_cap456_n41.json, resultados/null_aucs_real_n41.npy

# 2) Figuras §5.1.2 y §5.2.3 (requiere los dos ficheros del paso 1).
python src/figuras_cap5_cap6.py       # segundos
#    → resultados/fig_saturacion_percentiles_n41.png
#    → resultados/fig_permutacion_auc_n41.png

# 3) Modelos sociodemográfico y combinado (imprime AUC resustitución y LOO).
python src/modelos_sociodemo_n41.py   # < 1 min

# 3b) Matriz de intercorrelaciones de las dimensiones BEPE (Apéndice B / §5.1.1).
python src/intercorrelaciones.py      # segundos
#    → resultados/intercorrelaciones_bepe_n41.{csv,png}

# 3c) Figura 3 (§5.3.1): Thorndike observada vs corregida (lee el JSON del paso 1).
python src/fig_thorndike_obs_vs_corr.py   # segundos
#    → resultados/fig_thorndike_obs_vs_corr_n41.png
```

### Estudio de simulación (§5.4)

```bash
# 4) Calibración del generador (Paso 4): τ*, δ, betas por celda, tablas/figura real vs sintético.
python src/sim_capitulo7.py --calibrar        # ~90 s
#    → resultados/cap7_config_n41.json, cap7_calibracion_{tabla,betas}_n41.csv, cap7_calibracion_figura_n41.png

# 5) (Opcional) Piloto de tiempo / presupuesto de cómputo.
python src/sim_capitulo7.py --piloto          # ~5 min

# 6) Rejilla principal: 120 celdas × 500 réplicas, validación cruzada externa 5-fold.
python src/sim_capitulo7.py --rejilla --outer kfold --m 500    # ~3 h
#    → resultados/cap7_replicas_n41.csv

# 7) Fila de robustez con LOO (n=41), para contrastar con el 5-fold.
python src/sim_capitulo7.py --rejilla --outer loo --n-solo 41 --m 500   # ~3 h
#    → resultados/cap7_replicas_loo41_n41.csv

# 8) Refuerzo de validación: 500 nulas frescas independientes en las celdas señaladas.
python src/cap7_nulos_extra.py                # ~5 min
#    → resultados/cap7_nulos_extra_n41.csv

# 9) GATE — Validación del simulador (Paso 8.1): error tipo I y recuperación.
python src/sim_capitulo7.py --validar         # segundos

# 10) Productos (Paso 7): potencia del caso real, curva de aprendizaje, mapa de
#     detectabilidad y validación cruzada de Thorndike.
python src/sim_capitulo7.py --productos       # segundos
#    → cap7_celdas_n41.csv, cap7_potencia_caso_real_n41.csv,
#      cap7_curva_aprendizaje_n41.png, cap7_mapa_detectabilidad_n41.png,
#      cap7_thorndike_n41.{csv,png}

# 11) Análisis de sensibilidad (Paso 8.2): cópula, factores específicos,
#     tasa base 50/50, probit, mezcla de enseñas y estabilidad M=1000.
python src/sim_capitulo7.py --sensibilidad --m 500    # ~2,5 h
#    → cap7_sens_{betas,replicas}_n41.csv, cap7_sensibilidad_{resumen}_n41.csv,
#      cap7_sensibilidad_figura_n41.png
```

> Los pasos 6, 7 y 11 son los costosos. Pueden omitirse si se usan las réplicas
> ya incluidas en `resultados/`; los pasos 9–10 solo leen esos CSV.

## Reproducibilidad

Todas las semillas están fijadas (`SeedSequence` raíz `20260611` en la
simulación (§5.4), con ramas +100 rejilla, +200 LOO, +300 sensibilidad, +400 nulas
frescas; semilla `20260610` en `analisis_tfg.py`). El preregistro de decisiones
del estudio de simulación está en `resultados/cap7_preregistro.md`.

## Mapa salidas → memoria

Sección de la [memoria](resultados/TFG%20-%20Ord%C3%A1s%20Cernadas%2C%20Bernardo.pdf) en la que se cita cada salida de `resultados/`.

> Los nombres de los scripts y de las salidas (`sim_capitulo7.py`, `cap7_*`,
> `resultados_cap456_*`) conservan la numeración de una versión anterior de la
> memoria, que tenía nueve capítulos. No se han renombrado para no romper rutas;
> la correspondencia con la numeración actual es la de esta tabla.

| Sección | Producto |
|---|---|
| §5.1.1 / Apéndice B (intercorrelaciones BEPE) | `intercorrelaciones_bepe_n41.{csv,png}` |
| §5.1.2 — Figura 1 (efecto techo) | `fig_saturacion_percentiles_n41.png` |
| §5.2 (modelos) | `resultados_cap456_n41.json`, salida de `modelos_sociodemo_n41.py` |
| §5.2.3 — Figura 2 (permutación del AUC) | `fig_permutacion_auc_n41.png`, `null_aucs_real_n41.npy` |
| §5.3.1 — Figura 3 (Thorndike, obs. vs corregida) | `fig_thorndike_obs_vs_corr_n41.png` |
| §5.3.1 (Thorndike, contraste con el simulador) | `cap7_thorndike_n41.{csv,png}` |
| §5.3.2 — Figura 4 (curva de potencia frente a n) | `cap7_curva_potencia_n41.png` |
| §5.4.2 — Figura 5 (validación del generador) | `cap7_calibracion_{tabla,betas}_n41.csv`, `cap7_calibracion_figura_n41.png`, `cap7_validacion_tabla_n41.csv` |
| §5.4.3 — Figura 6 (potencia, curva de aprendizaje, detectabilidad) | `cap7_potencia_caso_real_n41.csv`, `cap7_curva_aprendizaje_n41.png`, `cap7_mapa_detectabilidad_n41.png` |
| §5.4.4 — Figura 7 (sensibilidad) | `cap7_sensibilidad_figura_n41.png`, `cap7_sensibilidad_resumen_n41.csv` |
