# 12 — Portal del cliente con asesor financiero virtual (Semana 19)

## 1. ¿Qué problema resuelve?

El asistente de la Semana 17 habla con **ejecutivos** sobre **la cartera**. Pero la meta del proyecto
también incluía al otro lado del mostrador: **un cliente que necesita orientación financiera y no tiene
a un ejecutivo disponible** (fuera de horario, en la app, o porque la sucursal está llena).

El portal del cliente (`http://localhost:8503`) es ese "ejecutivo virtual". En una sola pantalla,
el cliente ve:

| Sección | Qué ve | De dónde sale |
|---|---|---|
| **Tu salud financiera** | Puntaje de 0 a 100 y 4 pilares con semáforo (✓ / ! / ✕) | Reglas fijas sobre sus propios datos |
| **Tu plan de acción** | Pasos concretos, con montos, en orden de urgencia | Mismas reglas |
| **¿Y si recorto mis gastos?** | Control deslizante y presupuesto 50/30/20 | Cálculo en vivo |
| **Comparado con personas como tú** | Su lugar contra la mediana de su rango de edad | Grupos de al menos 30 clientes |
| **Pregúntale a tu asesor** | Chat con IA local (Ollama) | Sus datos + reglas de seguridad |

El dashboard ejecutivo **no cambia**: su asistente sigue ahí para la cartera.

## 2. Los 4 pilares (reglas públicas y explicables)

| Pilar | Cálculo | Bien | Atención | Urgente |
|---|---|---|---|---|
| Ahorro mensual | (ingreso − gasto) / ingreso | ≥ 20% (regla 50/30/20) | 0% a 20% | < 0% (gasta más de lo que gana) |
| Fondo de emergencia | ahorro acumulado / gasto mensual | ≥ 3 meses | 1 a 3 meses | < 1 mes |
| Deudas | mensualidad / ingreso | ≤ 36% (o sin préstamo) | 36% a 50% | > 50% |
| Historial crediticio | credit score | ≥ 670 | 580 a 669 | < 580 |

**Puntaje:** cada pilar vale 25 puntos (bien 25, atención 15, urgente 5). Si falta el dato de un
pilar, ese pilar no cuenta y el puntaje se reescala a 100. Niveles: **Sólida** (80 o más),
**En construcción** (55 a 79) y **Frágil** (menos de 55).

Las metas son **las mismas que usa el banco en sus KPIs** (`kpis_financieros.py`). Una prueba
automática (`test_metas_coinciden_con_los_kpis_del_banco`) falla si alguien cambia una y no la otra.

## 3. El asesor: misma arquitectura, otras reglas

```
pregunta ─► enrutador ─┬─► fraude / robo / NIP ──────────► "lo atiende una persona" (sin IA)
                       ├─► pregunta por OTRO cliente ────► "es confidencial" (sin IA)
                       └─► temas ─► consultas ─► LLM redacta ─► verificador ─► respuesta
                                    (sus datos)   (Llama 3.2)   (cifras, promesas,
                                                                 otros clientes)
```

**Temas que entiende:** salud financiera (resumen), ahorro, fondo de emergencia, deudas, credit score,
presupuesto 50/30/20, comparación con su grupo, **¿puedo pedir otro préstamo?** (capacidad de pago),
**¿me conviene invertir?** y **"no puedo pagar / perdí mi empleo"**.

**Reglas de seguridad (no dependen de la IA):**

1. **Fraude, robo, NIP o contraseñas:** se responde con un mensaje fijo que deriva a una persona y
   recuerda que el banco nunca pide el NIP. La IA ni siquiera ve la pregunta.
2. **Privacidad:** si la pregunta menciona a otro número de cliente, se niega. Si la respuesta de la IA
   menciona a otro cliente, el verificador la rechaza.
3. **Sin promesas:** si la IA escribe "te aprobamos" o "rendimiento garantizado", el verificador la
   rechaza y pide que se corrija.
4. **Capacidad de pago responsable:** una mensualidad nueva nunca se sugiere por arriba de **lo que le
   sobra al cliente cada mes** ni de la regla del 36%. A quien gasta más de lo que gana se le dice
   que hoy no le conviene endeudarse.
5. **Educación, no venta:** no recomienda productos de inversión específicos; dice si el cliente
   cumple los requisitos básicos (fondo de emergencia y deudas sanas) y lo envía con un asesor certificado.

Igual que el asistente ejecutivo: un reintento si la respuesta falla el control. Si vuelve a fallar,
se muestran **los datos exactos** (modo respaldo). Si Ollama está apagado, el portal funciona igual con
**modo básico**.

## 4. Decisiones de diseño (para defender ante el jurado)

| Decisión | Por qué |
|---|---|
| El cliente **no** ve su probabilidad de impago ni el índice de riesgo interno | Son herramientas internas del banco. Al cliente se le explica con **sus** números (ingreso, gasto, mensualidad, score) y reglas públicas que puede entender y mejorar |
| Comparaciones solo con **medianas de grupos de 30 o más** | Nadie puede deducir los datos de otra persona |
| El registro de preguntas guarda el **tema y el modo**, no el texto ni el número de cliente | Observabilidad sin guardar datos personales |
| Fraude y privacidad **sin IA** | Un modelo de 3B puede equivocarse; estos casos no admiten error |
| Acceso con número de cliente | Es una demostración. En producción sería el inicio de sesión de la banca en línea |

## 5. Cómo se usa

```bash
docker compose up -d --build streamlit-cliente     # la primera vez (o tras cambios)
# Portal del cliente: http://localhost:8503
```

En la pantalla de entrada hay **3 clientes de ejemplo** elegidos automáticamente: *Va por buen camino*,
*Gasta más de lo que gana* y *Deudas pesadas*. Son ideales para la demo porque muestran respuestas distintas.

**Guion de demo (1 minuto):**

1. Entra como *Gasta más de lo que gana*: puntaje Frágil, primer paso "Deja de gastar más de lo que ganas".
2. Mueve el control a 15% y muestra cómo cambia lo que le queda al mes.
3. Pregunta "¿Puedo pedir otro préstamo?": el asesor dice que hoy no le conviene, con la cifra.
4. Escribe "Me robaron la tarjeta": se deriva a una persona, sin IA.
5. Sal y entra como *Va por buen camino*: la misma pregunta da otra respuesta, porque son sus datos.

## 6. Evaluación (meta ≥ 84%)

```bash
docker compose exec streamlit-cliente python evaluar_asesor.py
```

Hace **10 preguntas a cada uno de los 3 clientes de ejemplo (30 en total)**: 8 temas más 2 casos
sensibles (robo y otro cliente). Mide enrutamiento, veracidad (cifras que existen en sus datos) y éxito
(la respuesta trae el dato correcto o, en los casos sensibles, se derivó). El resultado queda en
`data/reports/evaluacion_asesor.json` y alimenta el SLA **"Exactitud del asesor del cliente"** de la
Vista 0. En modo básico (sin IA) el banco da 100%; con IA, mide cuánto se pierde al redactar.

### 6.1 Primera medición con Llama 3.2 y mejoras

| Medición | Preguntas | Enrutamiento | Veracidad final | Éxito final | Tiempo promedio |
|---|---|---|---|---|---|
| 1a (19 nov) | 20 | 100% | 100% | **55%** | 20.4 s (la primera, 75 s) |

**Diagnóstico:** ninguna cifra inventada llegó al cliente (veracidad 100%), pero en 9 de 20 respuestas la IA dio
consejos correctos **sin la cifra que se preguntó**. Un modelo de 3B tiende a responder en general. Además, solo
hubo 2 clientes de ejemplo (20 preguntas), porque uno de los perfiles "de libro" no existe en los datos reales.

**Mejoras:**
1. **La cifra principal la escribe el sistema, no la IA.** La respuesta tiene dos partes: el "Dato principal",
   determinista y exacto, y 2–3 pasos concretos que redacta la IA. La IA aporta lo que hace bien (explicar y
   aconsejar) y ya no puede omitir el dato. El control de calidad sigue revisando los pasos (cifras, promesas,
   otros clientes).
2. **Siempre 3 clientes de ejemplo:** si un perfil "de libro" no existe, se usa el cliente más parecido.
3. **Velocidad:**
   - **Calentamiento:** el portal carga el modelo al abrirse, para que la primera pregunta no espere ~1 minuto.
   - **Modelo en memoria:** se queda cargado 60 min entre preguntas (`keep_alive`).
   - **Respuestas más cortas:** máximo 220 tokens, solo los pasos.
   - **Cifra al instante:** se muestra en cuanto se calcula, mientras la IA escribe.

> Nota honesta para el jurado: con este diseño el **éxito** mide que el cliente recibe el dato correcto (lo
> garantiza el sistema). La calidad de lo que redacta la IA se mide con la **veracidad** y con el
> porcentaje de respuestas que pasan el control a la primera.

## 7. Observabilidad

- `asistente_log.jsonl` ahora tiene el campo `canal` (`ejecutivo` / `cliente`). Las líneas antiguas,
  sin canal, cuentan como `ejecutivo`.
- Las preguntas **derivadas** (fraude, privacidad) no cuentan para el SLA "Asistente responde con IA",
  porque no pasar por la IA es lo correcto.
- Nuevo SLA `asesor_evaluacion` en `config/slas.yml` → ahora son **9 SLA**.

## 8. Archivos

| Archivo | Qué hace |
|---|---|
| `scripts/asesor_cliente.py` | Perfil, pilares, plan, comparativo, enrutador, consultas, verificador y `responder_cliente` |
| `scripts/evaluar_asesor.py` | Banco de 30 preguntas y métricas |
| `dashboard/cliente.py` | Interfaz del portal (puerto 8503) |
| `dashboard/data_loader.py` | Agrega `rango_edad` y `tipo_prestamo` al detalle por cliente |
| `tests/test_asesor_cliente.py` | 60 pruebas, sin Ollama |
| `docker-compose.yml` / `Dockerfile.streamlit` | Servicio `streamlit-cliente` |

## 9. Limitaciones

- Los datos son una **foto del día**: no hay historial mensual por cliente (solo el del cliente demo).
  Por eso el asesor no puede decir "este mes gastaste más que el anterior".
- El enrutador entiende **palabras clave**: preguntas muy rebuscadas caen en el resumen general.
- No hay inicio de sesión real ni canal de cita real: son las piezas que pondría el banco.
