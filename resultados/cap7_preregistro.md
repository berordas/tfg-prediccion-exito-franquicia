# Capítulo 7 — Preregistro de decisiones (Paso 0 del protocolo)

Fecha: 2026-06-11. Fijado ANTES de ejecutar la rejilla.

## Criterio de detección (único)
Una réplica detecta si su AUC_cv supera el percentil 95 de la distribución
nula empírica de su celda gemela d=0 (mismas n y τ, M=500). La potencia de la
celda es la fracción de réplicas que detectan, con IC binomial de Wilson al
95%. No se usa IC de DeLong ni ningún doble criterio.

## Rejilla experimental
- n ∈ {41, 60, 80, 120, 160, 200, 300, 500} (41 = caso real, obligatorio).
- d ∈ {0; 0,2; 0,35; 0,5; 0,8} (0 = calibración del error tipo I;
  0,35 = ancla principal, Cuesta et al. Tabla 7).
- τ ∈ {1; 0,5; τ* calibrada a la saturación real} (τ* resultante: ver
  cap7_config.json).
- M = 500 réplicas por celda (EE de potencia 0,80 ≈ 0,018). Nunca menos.
- Recorte autorizado solo por el extremo alto de n, previa consulta.

## Generador y mecanismo del criterio
- Bifactor a nivel de ítem, cargas de Cuesta et al. (2018) Tabla 2 (no se
  tocan). Discretización Likert 1-5 con umbrales comunes
  THR = [-2,1; -1,3; -0,55; 0,55]; PD = suma de 10 ítems.
- Grit: carga 0,45 en g y 0,45 en el específico de logro; mapeo afín a
  escala 1-5 (paso 0,5). Objetivo r(grit, ML) ≈ 0,63.
- Efecto verdadero en el factor general latente: P(y=1) = enlace(β₀ + β₁·g),
  enlace logit (probit en sensibilidad).
- **Interpretación preregistrada:** d se define como separación de g entre
  grupos en la POBLACIÓN GENERAL (los anclajes externos —Cuesta, Rauch &
  Frese— son poblacionales), verificada por esperanzas en megamuestra de
  10⁶. β₀ se calibra para tasa base 0,634 en la muestra SELECCIONADA de cada
  τ. Nota: la demo sim_tfg.py calibraba d post-selección; se registra la
  discrepancia de interpretación y se somete a confirmación del director.
- Preselección: top-τ del pool por la suma de las 8 PD observadas.
- Guardia de viabilidad: si la clase minoritaria de una réplica queda por
  debajo de 8 casos se regenera la réplica (se espera <1% de redibujos a
  n=41; el sesgo inducido se considera despreciable y se registra).

## Pipeline por réplica (Capítulo 5)
Logística elastic-net, 9 predictores (8 PD + grit), estandarización dentro
del pipeline, CV anidada: inner 5-fold GridSearchCV sobre
C ∈ {0,01; 0,1; 1} × l1_ratio ∈ {0; 0,5; 1} (solver saga), AUC validado.
**Desviación registrada y pendiente de aprobación:** bucle externo 5-fold
estratificado (semilla derivada) en lugar de LOO; LOO es inviable en la
rejilla (ver cap7_piloto.json). El dato real del Cap. 5 mantiene outer LOO.

## Semillas
SeedSequence raíz 20260611; spawn jerárquico (calibración, piloto +7,
rejilla +100) en orden determinista de celdas (τ exterior, d, n) y réplicas.
Las semillas por réplica quedan registradas en cap7_replicas.csv.

## Qué se reporta pase lo que pase
1. Validación 8.1 (tipo I en todas las celdas d=0, con control split-half
   adicional; recuperación en n=500, d=0,8, τ=1 con AUC ≈ Φ(0,8/√2) = 0,714).
2. Tabla de potencia del caso real (n=41, τ*) por d con IC de Wilson.
3. Curva de aprendizaje (mediana e IQR de AUC_cv frente a n, por d, τ*) y
   n al que la potencia cruza 0,80 para d = 0,35.
4. Mapa de detectabilidad n×d por τ con el caso real marcado.
5. Validación cruzada Thorndike (simulador vs fórmula del Cap. 6).
6. Contraste resustitución vs validado por celda (sobreajuste vs n).
7. Sensibilidad (8.2): cópula gaussiana, efecto en específicos (ST+AM),
   tasa base 50/50, probit, y 1.000 réplicas en celdas críticas.

## Desviaciones registradas tras la calibración (v2, 2026-06-11)
1. **Selección ruidosa en lugar de truncamiento duro.** El top-τ duro sobre
   la suma BEPE es incompatible con la muestra real (desplazada ~+1 SD pero
   con U de Thorndike ≈ 1 e intercorrelación 0,52): el truncamiento duro
   colapsa SD e intercorrelación. Se usa v = ρ·z(suma) + √(1−ρ²)·ε.
   Calibrado: τ* = 0,9, ρ = 0,2 (selección débil).
2. **Pool de candidatos con perfil propio.** δ_g = 0,75 sobre el factor
   general (ancla: autónomos vs empleados de Cuesta, MANOVA d≈0,35 global)
   más desplazamientos residuales por específico ajustados a las 8 medias
   reales (AE +2,46, TR +1,44, IN +1,10, OP +0,65, ML +0,39, LI +0,07,
   TE −0,15, AU −0,81 — franquiciados altos en todo menos autonomía).
   El panel τ=1 queda como población limpia (sin perfil ni selección).
3. **Cortes de saturación** recuperados de la tabla normativa real vía los
   pares PD/Pc del dataset (menor corte consistente: max PD con Pc<95, +1).
4. **Outer 5-fold estratificado** en la rejilla (LOO inviable: 973 h
   secuenciales vs 24 h; ver cap7_piloto.json). Opcional: fila n=41 con LOO
   como robustez.
5. **d definido en población general** (anclas externas), no post-selección
   como en la demo sim_tfg.py.

## Decisiones confirmadas por el alumno (2026-06-12, antes de lanzar)
1. Rejilla completa con outer 5-fold estratificado (presupuesto ~2 h en 16
   núcleos), 120 celdas × 500 réplicas, sin recortar n. Nota: el encargo
   escribió "5-fold estratificado repetido"; lo presupuestado y preregistrado
   es UNA pasada de 5-fold estratificado (semilla derivada) por réplica — un
   outer repetido multiplicaría el presupuesto confirmado; queda registrado.
2. Calibración ACEPTADA con los 12 fallos residuales documentados. Las SD
   seleccionadas quedan comprimidas (−15 a −25%): esa dirección del sesgo
   solo puede INFLAR la potencia estimada, de modo que la conclusión de
   infrapotencia se sostiene a fortiori (estimación conservadora). La mezcla
   Brooklyn/Japanese NO entra en el generador principal; va a la
   sensibilidad (Paso 8.2) como variante.
3. Fila de robustez LOO en n=41 (15 celdas × 500, semilla raíz 20260611+200,
   archivo cap7_replicas_loo41_n41.csv) para mostrar que el caso real no
   depende del bucle externo.
4. Gate obligatorio: validación 8.1 ANTES de cualquier producto; si tipo I
   (d=0) o la recuperación (n=500, d=0,8, τ=1, AUC≈0,714) fallan, se detiene
   el análisis y no se generan productos.

## Diseño preregistrado de la sensibilidad (Paso 8.2)
Rejilla reducida por variante: n ∈ {41, 80, 120, 200, 300, 500} ×
d ∈ {0; 0,35; 0,8} × panel τ* (M=500; nula empírica propia de cada variante
y cada n). Variantes: (a) cópula gaussiana sobre la matriz de correlación y
marginales empíricas de la muestra real, efecto en el primer componente
latente normalizado (sin mecanismo de selección: la cópula reproduce
directamente la población seleccionada); (b) efecto en factores específicos
(tolerancia al estrés + motivación de logro, (s_ST+s_AM)/√2) en vez de g;
(c) tasa base 50/50; (d) enlace probit; (e) mezcla de dos enseñas
(desplazamientos por específico ajustados a las medias de Brooklyn n=23 y
Japanese n=18, pesos 23/41 y 18/41); (f) estabilidad M=1000 en las celdas
críticas (n=41, τ*, los tres d). Semilla raíz 20260611+300.

## Tolerancias de calibración (Paso 4, fijadas ex ante)
±10% relativo en: medias y SD de las 9 puntuaciones, intercorrelación media
(~0,52), saturación Pc≥95 por dimensión (TR ~49%, AE/IN ~34-36%), media de
grit (~4,44) y r(grit, ML) (~0,61). Las métricas que fallen se documentan y
solo se ajustan umbrales de discretización, parámetros del grit o τ — nunca
las cargas publicadas.
