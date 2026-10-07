# DMEyF 2026 · Primera Competencia: predicción de bajas del Paquete Premium

Ezequiel Abregu · Data Mining en Economía y Finanzas (UBA), comisión de los lunes.

Objetivo: elegir a qué clientes estimular en **`202108`** para retener a los que van a ser `BAJA+2`, maximizando la ganancia: +$1.072.500 por cada `BAJA+2` estimulado y −$27.500 por cada estimulado que no lo es.

**Entrega final:** `entrega_final/final01fe03x40N12000/final01fe03x40N12000_locro.csv`, con 12.000 clientes. Es el archivo que se subió al bot de Zulip, como `submit final01fe03x40N12000`. Dio **110,25** en el leaderboard público del bot, que mide ~25 % de agosto.

---

## 1. El modelo final, en una tabla

| Pieza | Elección | Dónde se decidió |
|---|---|---|
| Datos | `competencia_01_crudo.csv` → clase ternaria con horizonte de 2 meses (lógica de la cátedra) | `01` |
| Clase | **`BAJA+1` + `BAJA+2` contra `CONTINUA`**. La ganancia siempre se mide solo con `BAJA+2` | `02` |
| Limpieza por drift (**V5**) | Sin `Master/Visa_fultimo_cierre` (rotas en `202106`) ni `Master/Visa_Finiciomora` (PSI de 3 a 6). `ccajas_depositos` → `NA` en `202105` (todo en cero ese mes). **Los préstamos personales se quedan**: sacarlos ganaba en junio, pero en el público de agosto restó 4,5 | `04` |
| Variables nuevas | **B2** historia del TP4 (lags, delta, promedio, mínimo, máximo y pendiente de 3 meses de 8 variables) · **B3** señales del análisis de segmentación (empieza a pagar la comisión, historia del sueldo, consumo de tarjeta contra sus meses anteriores…) · **B4** los 73 montos pasados a ranking dentro del mes · **B5** lag 1 y delta 1 de todas las variables. En total, **507 variables** | `05` |
| Hiperparámetros | LightGBM `gbdt`, **prueba 42 de Optuna**: 36 hojas, `min_data_in_leaf` 1.855, `learning_rate` 0,042, 637 árboles, `feature_fraction` 0,40, `bagging_fraction` 0,93 con `bagging_freq` 1 | `08` |
| Entrenamiento final | `202103`–`202106` (654.066 filas). `202107` no se usa: no tiene `BAJA+2` | `03` |
| Ensamble | Promedio de las probabilidades de **40 semillas**: dos juegos de 20 primos de 6 dígitos, sacados de la semilla primigenia **979201** | `03` + `07` |
| Corte | Los **12.000** clientes de mayor probabilidad. N se eligió con el leaderboard público: 11.500–12.500 es la meseta | `07` |

## 2. Cómo reproducir la entrega final

### Requisitos

- Python 3.14 con las versiones de `requirements.txt` (`pip install -r requirements.txt`): `duckdb` 1.5.5, `lightgbm` 4.7.0, `optuna` 4.9.0, `pandas` 2.2.2, `numpy` 2.4.3, `scipy` 1.17.1, `scikit-learn` 1.8.0, `matplotlib` 3.10.8, `jupyter`.
- **El dataset original** `competencia_01_crudo.csv` en `../dataset/` relativo a la raíz de este repo. Si está en otro lugar, se cambia `CRUDO` en la celda de parámetros del `01`: es la única ruta externa.
- ~16 GB de RAM: el dataset con las variables nuevas ocupa ~2,5 GB en memoria.

### Camino corto: solo lo que entra en el archivo final (~1 hora)

Los notebooks están en `notebook/`. Se corren **en este orden y enteros**. En cada paso figura el bloque completo de parámetros; lo que no está en el bloque queda como viene.

1. **`01_pipeline_base`**, sin cambios. Crea `data/competencia_01_base.parquet` y controla las filas, la clase y los IDs (~1 min).
2. **`02_linea_base`** con `MODO_RAPIDO = True`. Escribe las semillas (`semillas.json`) y las configuraciones que usan los demás notebooks (~2 min).
3. **`04_drift`** con `MODO_RAPIDO = True`. Escribe `variantes.json`, que incluye **V5** (~1 min).
4. **`05_variables`**, para armar el parquet de variables nuevas y la lista de variables (`bloques.json`):
   ```python
   MODO_RAPIDO = True
   BASE = 'V5'
   CONFIG = 'z470_bajas'
   BLOQUES_A_CORRER = []
   COMBINACIONES = [['B2', 'B3', 'B4', 'B5']]
   REFERENCIA = None
   ```
   Tarda ~5 min.
5. **Los hiperparámetros:** `exp/08_optuna/hiperparametros_p42.json` viene en el repo. Para rehacer la búsqueda, ver el camino completo.
6. **`03_entrega`, juego de semillas 1** (~21 min):
   ```python
   CONFIG = 'z470_bajas'
   VARIANTE = 'V5'
   N_ENVIOS = 12_000
   BLOQUES = 'B2+B3+B4+B5'
   NOMBRE_SUBMIT = 'fe03z470bajasV5B2345Opt42'
   HIPER_OPTUNA = 42
   JUEGO_SEMILLAS = 1
   MESES_FINAL = [202103, 202104, 202105, 202106]
   ```
7. **`03_entrega`, juego de semillas 2** (~21 min). Mismo bloque, con dos cambios:
   ```python
   NOMBRE_SUBMIT = 'fe03z470bajasV5B2345Opt42J2'
   JUEGO_SEMILLAS = 2
   ```
8. **`07_ensamble`**, que promedia las 40 semillas y corta en 12.000 (segundos):
   ```python
   ORIGENES = ['fe03z470bajasV5B2345Opt42', 'fe03z470bajasV5B2345Opt42J2']
   N = 12_000
   NOMBRE = 'final01fe03x40N12000'
   ENSAMBLES, MODELOS_POR_ENSAMBLE = 20, 20
   SEMILLA_SORTEO = 979201
   ```
9. **Control:** el archivo generado tiene que ser **idéntico** al entregado.
   ```bash
   shasum -a 256 entregas/final01fe03x40N12000/final01fe03x40N12000_locro.csv
   # 88f49493daa9598f89db702ce72b61f2d64df189301cfeecb6bf6d2f86259954  (entrega_final/final01fe03x40N12000/*.sha256)
   ```

**Por qué alcanza con el modo rápido en el `02`, el `04` y el `05`:** esos notebooks escriben sus configuraciones (semillas, variantes, lista de variables) sin importar el modo. El modo completo solo agrega los backtests con 10 semillas que sirvieron para decidir. Los controles internos verifican que las semillas, las variables y el orden de las filas coincidan en cada paso.

### Camino completo: todos los experimentos (~4–5 horas)

1. **`01`**.
2. **`02`, `04` y `05` con `MODO_RAPIDO = False`.** En el `05`: `BASE = 'V5'` y las `COMBINACIONES` que vienen por defecto, más `['B2', 'B3', 'B4', 'B5']`.
3. **`08_optuna`** con `MODO_RAPIDO = False`, `N_PRUEBAS = 50` y `ELEGIDA = 42`: 50 pruebas, la validación y `hiperparametros_p42.json`.
4. **Los pasos 6 a 9 del camino corto.**

**Aviso:** Optuna usa un sampler con semilla (TPE, 979201), pero LightGBM puede dar diferencias numéricas mínimas entre máquinas, y eso puede cambiar el camino de la búsqueda. Por eso el repo incluye el resultado (`hiperparametros_p42.json`), y el camino corto parte de ahí.

## 3. Qué hace cada notebook

| Notebook | Qué hace |
|---|---|
| `01_pipeline_base` | Crudo → clase ternaria en SQL (DuckDB) → parquet base. Tabla de calidad por mes (nulos y ceros) |
| `02_linea_base` | Línea base: hiperparámetros de la cátedra (`z601`, `z470`) × las dos clases, backtests A (`202103`–`04` → `202106`) y B (`202103` → `202105`), 10 semillas, ganancia en la meseta suavizada (501) |
| `03_entrega` | Modelo final con 20 semillas sobre `202103`–`202106`, predicción de `202108`, CSV por semilla y semillerío, checklist del formato del bot y registro del submit |
| `04_drift` | Variantes de limpieza por drift (V1–V5), comparación pareada con Wilcoxon |
| `05_variables` | Bloques de variables nuevas (B1–B5) y sus combinaciones, contra la base y contra una referencia; prueba C (`202103`–`05` → `202106`) para la historia |
| `07_ensamble` | Promedio de las probabilidades de varios submits del `03`, cortado en N, en un solo archivo |
| `08_optuna` | Optuna en A con el 20 % de los `CONTINUA`, validación con todos los datos y 5 semillas en A y C, contra `z470` |


## 4. Lo que más movió la ganancia (público)

| Paso | Público |
|---|---:|
| Línea base (`z470`, las dos bajas, semillerío) | 88,30 |
| + limpieza V5 + variables B2, B3 y B4 (semillerío, N = 9.500) | 98,01 |
| N = 12.000 en lugar de 9.500 (mismo modelo, 20 semillas) | 94,86 → 101,54 |
| + B5 (20 semillas) | 102,99 |
| + hiperparámetros de Optuna, prueba 42 (20 semillas) | 103,62 |
| **+ semillerío de la prueba 42 (20 semillas / 40 semillas)** | **110,41 / 110,25** |

Dos lecciones:
1. **El público con 20 semillas sirvió como segundo juez.** Rechazó una limpieza que los backtests de junio adoptaban, y mostró que el N de los backtests era corto para agosto.
2. **El ensamble rinde más cuanto más varían las semillas.** Con `z470`, el semillerío sumaba +1 a +3. Con la prueba 42 (bagging y `feature_fraction` de 0,40), sumó +6,8.

## 5. Estructura

```
notebook/                              los notebooks 01–05, 07 y 08
requirements.txt                       las versiones de las librerías
exp/08_optuna/hiperparametros_p42.json los hiperparámetros del modelo final
entrega_final/final01fe03x40N12000/    el CSV entregado (el que se subió al bot) y su sha256
```

Al correr, los notebooks crean `data/` (los parquets), `exp/` (curvas, probabilidades y configuraciones) y `entregas/` (los CSV de cada corrida y un registro). No se versionan: se regeneran.
