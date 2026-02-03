# Plasticidad Adaptativa Conversacional

Este proyecto simula la plasticidad adaptativa en una conversación, inspirada en principios de neurociencia y aprendizaje automático. Se utilizan métricas cuantitativas para modelar la capacidad de adaptación de un sistema (IA o humano) ante la novedad y la sorpresa en el diálogo.

## Variables Clave (Operacionalizadas)

- **C_{n+1} (Plasticidad Adaptativa):**
  - Métrica agregada de capacidad de adaptación.
  - En IA: proxy de actualización de pesos tras nuevos datos.
  - En humanos: cambio en conectividad cerebral (fMRI).

- **σ(A·E_n) (Modulador de Atención):**
  - σ: función sigmoide.
  - A: factor de interacción (ratio tokens nuevos, función continua).
  - E_n: energía de error predictivo (perplejidad promedio por token).

- **∑(R_t·D_KL) (Integración Evolutiva de Novedad):**
  - R_t: memoria activa (decaimiento exponencial).
  - D_KL: divergencia KL entre distribuciones de probabilidad antes y después del nuevo input.

## Mejoras implementadas

- Perplejidad normalizada por token.
- D_KL estable (clip y normalización de probabilidades).
- Factor A como función continua.
- R_t con decaimiento exponencial.
- Guardado de métricas por interacción (JSON).

## Uso

1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta el script:
   ```bash
   python Untitled-1.py
   ```
3. Interactúa en consola. Al finalizar, se guardan métricas en `metricas_interaccion.json`.

## Roadmap

- [x] Feedback conceptual y operacionalización de variables
- [x] Parches de optimización y robustez
- [x] README y requirements.txt
- [ ] Novedad semántica (embeddings, cosine distance)
- [ ] Experimentos: prompts repetitivos vs novedosos
- [ ] Visualización de métricas

## Referencias
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
- Vaswani, A. et al. (2017). Attention is all you need.
