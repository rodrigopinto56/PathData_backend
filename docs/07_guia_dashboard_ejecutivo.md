# 07 — Guía del dashboard ejecutivo

*Para quien usa el tablero, no para quien lo programa.*

**Dirección:** http://localhost:8502 · Se actualiza solo cada día.

## Qué responde, en el orden en que se lee

| Sección | Pregunta | Cómo leerla |
|---|---|---|
| **Diagnóstico** (recuadro superior) | ¿Cómo está la cartera hoy? | Una etiqueta (✓ Saludable · ! Requiere atención · ✕ Crítico) y una frase con lo que está fuera de meta. |
| **Indicadores clave** | ¿Cumplimos nuestras metas? | 4 tarjetas, cada una con su valor, su meta y una explicación en palabras. |
| **¿Cómo están nuestros clientes?** | ¿Cuántos clientes están en riesgo? | Barra verde / amarilla / roja con el porcentaje y el número de clientes de cada nivel. |
| **Dinero en juego** | ¿Cuánto prestamos y cuánto podríamos perder? | Dinero prestado, pérdida esperada ("por cada $100 prestados se esperan perder $X") y qué parte está en clientes riesgosos. |
| **¿Dónde está el riesgo?** | ¿Qué grupo de clientes preocupa más? | Elige "Ver por" (región, calificación crediticia, edad…). La línea punteada es la meta (40); barras a la derecha de la línea = fuera de meta. |
| **Acciones sugeridas** | ¿Qué hacemos? | Lista priorizada (Alta / Media) que se genera sola a partir de lo que está fuera de meta. |
| **Clientes prioritarios** | ¿A quién llamamos primero? | Clientes en rojo, del más al menos riesgoso. Se puede descargar la lista completa en Excel (CSV). |

## Los 4 indicadores clave

| Indicador | Qué significa | Meta |
|---|---|---|
| **Capacidad de ahorro** | De cada $100 que gana el cliente típico, cuánto le queda después de sus gastos. El *rango confiable* dice entre qué valores está el dato real. | 20% o más |
| **Endeudamiento mensual** | De cada $100 de ingreso, cuánto se va a pagar deudas (clientes con préstamo). | Menos de 36% |
| **Índice de riesgo** | Calificación de 0 (muy sano) a 100 (muy riesgoso) que combina endeudamiento, probabilidad de impago, ahorro e historial crediticio. | Menos de 40 |
| **Estabilidad de ingresos y gastos** | Si el ingreso y el gasto del cliente cambian mucho mes a mes. | *Pendiente*: requiere historial mensual |

## Semáforo de clientes

| Color | Índice de riesgo | Qué hacer |
|---|---|---|
| ✓ Verde | menos de 40 | Seguimiento normal |
| ! Amarillo | 40 a 69 | Vigilar; ofrecer productos preventivos (ahorro, asesoría) |
| ✕ Rojo | 70 o más | Contactar; evaluar reestructura o ajuste de plazo |

## Compartir

- **Reporte de una página:** botón *Descargar reporte ejecutivo*. Es un archivo que se abre en cualquier navegador y se imprime o guarda como PDF (Ctrl + P).
- **Reporte diario automático:** el proceso de datos publica el mismo reporte cada día en `gold-layer/reportes_ejecutivos/` (MinIO) y en `data/reports/ejecutivo/`.
- **Fecha de corte:** el selector arriba a la derecha permite consultar días anteriores.

## Avisos importantes

- Los datos son **de demostración** (dataset público sintético); los valores ilustran el funcionamiento, no la cartera real.
- Si aparece *"se calcularon con una versión anterior del proceso"*, el equipo de datos debe volver a ejecutar el proceso diario para esa fecha.
