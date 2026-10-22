# 09 — Simulador financiero Monte Carlo (Semana 16)

Código: `scripts/simulador_montecarlo.py` · Configuración: `config/simulador.yml` ·
Pruebas: `tests/test_simulador_montecarlo.py` · Orquestación: tarea `simular_escenarios` de `transformacion_gold_dag` ·
Dashboard: sección **"Simulador de escenarios"** del tablero ejecutivo.

## 1. ¿Qué es Monte Carlo?

En vez de calcular **un** resultado ("la pérdida esperada es \$2.3 M"), se simulan **miles de futuros posibles**
con reglas de probabilidad y se mira la **distribución** completa: qué pasa en promedio y, sobre todo, qué tan
malo puede ponerse y qué tan seguido.

El nombre viene del casino de Monte Carlo (Mónaco): la técnica se apoya en el azar, como una ruleta. La bautizaron
así en los años 40 los científicos del proyecto Manhattan (Ulam y von Neumann). En banca es la forma estándar de
medir el **riesgo de cola**: los años raros pero muy caros que el promedio esconde.

| Pregunta de negocio | Sin Monte Carlo | Con Monte Carlo |
|---|---|---|
| ¿Cuánto vamos a perder? | Un número: la pérdida esperada | La distribución: esperada, 1 de 100, 1 de 1,000 |
| ¿Nos alcanza el colchón? | No se puede responder | "Se rebasa en X% de los años" |
| ¿Y si hay crisis? | Se recalcula a mano | Escenarios de estrés y sliders en el dashboard |

## 2. Simulador de la cartera de crédito (modelo de Vasicek)

Es el modelo de **un factor** que usa Basilea II/III para el capital regulatorio de crédito.

Cada simulación es **un año posible**:

1. Se sortea un **factor económico común** `Z ~ N(0,1)`: qué tan buena o mala es la economía ese año.
2. Cada préstamo `i` cae en impago si `√ρᵢ · Z + √(1−ρᵢ) · εᵢ < Φ⁻¹(PDᵢ)`.
   - `εᵢ ~ N(0,1)` es la "suerte propia" del cliente.
   - `PDᵢ` es la probabilidad de impago del **modelo de la Semana 15** (`riesgo_score` en `fact_posicion_financiera`).
3. Pérdida del año = `Σ EADᵢ × LGD` de los préstamos que cayeron. EAD es el monto del préstamo.

La **correlación ρ** es la clave: mide cuánto dependen los impagos de la economía común. Con ρ = 0 los impagos
son independientes y casi nunca hay años catastróficos. Con ρ > 0, en un año malo caen **muchos a la vez**.
Por defecto se usa la fórmula de Basilea para crédito minorista: 16% para PD bajas, que baja hasta 3% para PD altas.

### Escenarios (`config/simulador.yml`)

| Escenario | PD | LGD | Correlación |
|---|---|---|---|
| Base | la del modelo | 45% (estándar Basilea) | Basilea |
| Adverso | × 1.5 | 55% | Basilea |
| Severo | × 2.5 | 65% | 25% fija (más contagio) |

### Resultados (`gold_simulacion_cartera`)

| Campo | Significado |
|---|---|
| `perdida_esperada` | Promedio de los años simulados; debe cuadrar con `Σ PD × EAD × LGD` (se prueba) |
| `var_95`, `var_99`, `var_99_9` | **Value at Risk**: pérdida que solo se supera en 5%, 1% y 0.1% de los años |
| `expected_shortfall_99` | Promedio de las pérdidas **dentro** del peor 1% (qué tan malo es lo malo) |
| `capital_economico` | `VaR 99.9% − pérdida esperada`: lo que hay que tener **además** de las reservas |
| `prob_superar_colchon_pct` | % de años en que la pérdida rebasa el **colchón actual** (VaR 99.9% del escenario base = reservas + capital) |
| `prob_superar_reservas_pct` | % de años en que la pérdida rebasa la pérdida esperada base (solo reservas) |

`gold_simulacion_cartera_distribucion` guarda el histograma de cada escenario (40 rangos).

> **Nota de lectura.** `prob_superar_reservas_pct` puede ser **menor** en el escenario severo que en el adverso.
> Más correlación concentra las pérdidas en pocos años catastróficos y deja muchos años "normales". Por eso
> el tablero ejecutivo muestra el **colchón** (reservas + capital), que sí crece siempre con la severidad.

## 3. Simulador del cliente demo (bootstrap)

Cada simulación arma **12 meses futuros** sorteando, con reemplazo, **meses reales** de los últimos 24 meses de su
historial. El ingreso y el gasto de un mismo mes se sortean **juntos**, para conservar su relación. Después
aplica el escenario y sigue el saldo ahorrado:

`saldo(mes) = saldo_inicial + Σ (ingreso × (1 + Δingreso) − gasto × (1 + Δgasto))`, con ingreso 0 los primeros
`meses_sin_ingreso`.

| Escenario | Supuesto |
|---|---|
| Base | Se repite su comportamiento de los últimos 24 meses |
| Inflación | Gastos +10% |
| Recesión | Ingresos −20% |
| Desempleo | Sin ingreso los primeros 3 meses |

Resultados:

- `gold_simulacion_cliente_demo`: abanico por mes con percentiles 5/25/50/75/95 y % de futuros con saldo negativo.
- `gold_simulacion_cliente_demo_resumen`: probabilidad de quedarse sin ahorro, mes típico en que se agota,
  saldo final mediano y flujo mensual promedio.

La serie mensual se lee de `gold_pronostico_cliente_demo` (Semana 15): es la misma que ve el dashboard, sin volver a leer Silver.

## 4. Dónde corre

- **DAG (oficial, diario):** `construir_gold → pronosticar_cliente_demo → simular_escenarios → calcular_kpis`.
  - Usa 10,000 años por escenario en la cartera y 5,000 futuros por escenario en el cliente.
  - Con 17,840 préstamos (tamaño real) tarda unos 13 s. Escribe `_simulacion_metadata/<fecha>.json` con la configuración usada.
- **Dashboard (interactivo):** sliders de PD ×, LGD y contagio para la cartera (2,000 años en vivo, ~1 s), y de
  cambio de ingreso/gasto, meses sin ingreso y ahorro inicial para el cliente.
  - Usa el mismo código y el mismo YAML que el DAG: se copian a la imagen en `Dockerfile.streamlit`.
- **Reproducible:** semilla fija (`semilla: 42`). La misma fecha y configuración dan el mismo resultado.

## 5. Validación (`tests/test_simulador_montecarlo.py`, 11 pruebas)

- La pérdida esperada simulada cuadra con `Σ PD × EAD × LGD` (±3%).
- En una cartera grande y homogénea, el **VaR simulado coincide con la fórmula cerrada de Vasicek/Basilea**:
  `Φ((Φ⁻¹(PD) + √ρ Φ⁻¹(q)) / √(1−ρ))` (±6%).
- Sin correlación no hay colas gordas; con correlación, el VaR 99% es varias veces la media.
- Los escenarios quedan ordenados: base < adverso < severo, en pérdida esperada, VaR y expected shortfall.
- Se cumplen `EL ≤ VaR95 ≤ VaR99 ≤ VaR99.9` y `ES ≥ VaR`. El colchón base se rebasa ~0.1% de los años, por definición.
- Cliente: el abanico está ordenado, cada escenario de estrés es peor que el base y el bootstrap conserva el flujo promedio.
- **Prueba de punta a punta** (S3 simulado, transacciones reales): la pérdida esperada simulada (5.12% de la
  exposición) cuadra con el KPI `perdida_esperada_pct_exposicion` calculado por fórmula (5.10%).

## 6. Supuestos y limitaciones (decirlas siempre)

- **Ahorro inicial del cliente:** el dataset no lo trae. Se supone 3 meses de su gasto promedio y se puede ajustar en el dashboard.
- **El cliente demo gasta más de lo que gana:** en los últimos 24 meses, unos −\$6,800 al mes. Por eso, aun en el
  escenario base, agota su ahorro en ~79% de los futuros. Es un dato del dataset, coherente con el pronóstico de la Semana 15.
- **El bootstrap no conserva la estacionalidad** (diciembre puede caer en cualquier mes) ni tendencias. Para 12 meses
  es aceptable; un modelo de series de tiempo con choques sería la mejora siguiente.
- **Cartera:**
  - LGD fija por escenario; en la realidad varía por garantía.
  - EAD = monto original del préstamo, sin amortización.
  - Horizonte de 1 año.
  - Las PD vienen del modelo de la Semana 15, que se entrenó con otra fuente (ver [06](06_decisiones_y_deuda_tecnica.md)).
- Los escenarios (×1.5, ×2.5) son **supuestos de negocio**, no un pronóstico macroeconómico. Se ajustan en el YAML sin tocar código.
