# Etapa 2: de la segmentacion al diagnostico (ACDC original)

La etapa 1 (`cardiac_dataset.py`, `train.py`) segmenta solo el miocardio con los `.npy` del curso. La etapa 2 usa el ACDC original para llegar a un diagnostico:

```
resonancia (cada corte en 2D)
  -> U-Net multiclase: fondo, cavidad del VD, miocardio, cavidad del VI
  -> volumenes en ml con el espaciado real de cada paciente
  -> biomarcadores: FE del VI y del VD, volumenes, masa, grosor de pared
  -> clasificador: NOR, MINF, DCM, HCM, RV
  -> modulo de seguridad: incertidumbre por paciente, para marcar casos dudosos
```

La red sigue siendo 2D, como pide el enunciado: cada corte se segmenta por separado y solo despues los cortes se apilan para medir volumenes.

## Datos (verificados el 2026-10-14)

- `data/ACDC/database/{training,testing}/patientNNN/`: 100 + 50 pacientes, 20 + 10 por grupo. Todos tienen anotacion en ED y ES.
- Licencia CC BY-NC-SA 4.0: uso no comercial, citar Bernard et al. 2018.
- **Etiquetas del ACDC: 0 fondo, 1 cavidad del VD, 2 miocardio, 3 cavidad del VI.** Son distintas de los `.npy` de la etapa 1, donde 1 es miocardio y 2 la cavidad del VI. Lo confirma la fisiologia: las etiquetas 1 y 3 se achican de ED a ES en el 100 % de los pacientes y el miocardio (2) no.
- Espaciado en el plano entre 1.37 y 1.68 mm; entre cortes, 5 o 10 mm; de 6 a 21 cortes. Las intensidades vienen en float32, int16 o uint8 segun el equipo.
- Medianas por grupo con las mascaras reales: el clasificador tiene senal real.

| grupo | VI FD (ml) | FE VI (%) | VD FD (ml) | FE VD (%) | masa (g) |
|---|---|---|---|---|---|
| NOR | 128 | 62 | 147 | 51 | 101 |
| MINF | 179 | 31 | 128 | 52 | 133 |
| DCM | 281 | 17 | 171 | 31 | 168 |
| HCM | 142 | 70 | 126 | 61 | 161 |
| RV | 123 | 56 | 212 | 39 | 86 |

## Preprocesamiento

1. Normalizacion **por volumen**: recorte a los percentiles 1-99 y z-score. Una normalizacion global como la de la etapa 1 no sirve, porque cada equipo usa otra escala.
2. Remuestreo de cada corte a 1.5 mm por pixel: bilinear para la imagen, `nearest-exact` para la mascara.
3. Recorte central de 192x192. El peor caso del ACDC necesita 179 px, asi que el recorte no corta el corazon.
4. La prediccion vuelve a la grilla nativa (recorte inverso y probabilidades interpoladas) antes de medir: los volumenes se miden con el espaciado original.

## Protocolo de evaluacion (contra fugas)

- **Las divisiones son por paciente, nunca por corte**: los cortes de un mismo corazon se parecen.
- Los 50 pacientes de `testing` no se tocan hasta la evaluacion final.
- Segmentacion: validacion cruzada de 5 pliegues sobre los 100 de `training`, estratificada por grupo (4 por grupo en cada pliegue).
  - Numero fijo de epocas y se guarda el ultimo checkpoint. El pliegue retenido solo se monitorea: elegir la epoca con el pliegue retenido contaminaria las predicciones fuera de pliegue.
- **Predicciones fuera de pliegue**: cada paciente de training se segmenta con el modelo que no lo vio. El clasificador se entrena con biomarcadores de esas predicciones, no de mascaras que la red memorizo; si no, aprende con medidas mas limpias que las que vera en test.
- Test: ensamble de los 5 modelos.
- Cota superior ("oraculo"): el mismo clasificador con biomarcadores de las mascaras reales. La distancia al oraculo mide cuanto cuesta segmentar automaticamente.

## Modulo de seguridad

- Incertidumbre por *test-time augmentation*: las 8 simetrias del cuadrado (flips + rotaciones de 90 grados) con el mismo modelo. Se usa la misma definicion en las predicciones fuera de pliegue (un modelo por paciente) y en test (el promedio sobre los 5 modelos).
- Puntaje por paciente: dispersion de los biomarcadores entre vistas (por ejemplo FE = 45 +- 6 %) y margen de probabilidad del clasificador.
- El umbral de alerta se fija con training (fuera de pliegue) y se verifica en test: los pacientes marcados deberian concentrar los errores. Se reporta con una curva riesgo-cobertura.

## Metricas

- Segmentacion: Dice e IoU por clase y por paciente (3D, estandar ACDC), mas el IoU 2D del miocardio (la metrica del curso).
- Biomarcadores: sesgo y limites de acuerdo de Bland-Altman y correlacion contra las mascaras reales.
- Diagnostico: exactitud y matriz de confusion 5x5; pipeline contra oraculo.
- Seguridad: exactitud en los no marcados contra los marcados; riesgo-cobertura.

## Etapas de implementacion

1. `acdc_data.py`: lectura, preprocesamiento, cache, Dataset y pliegues, con tests.
2. Perdida multiclase y `train_acdc.py` por pliegue; entrenar los 5 pliegues en GPU.
3. Inferencia con TTA y vuelta a la grilla nativa; `biomarcadores.py` con tests sobre geometrias de volumen conocido.
4. `diagnostico.py`: clasificador, oraculo y evaluacion en test.
5. Modulo de seguridad y figuras en `outputs/acdc/`.
