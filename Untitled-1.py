import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import kl_div

# ===========================

# ===========================
# CONFIGURACIÓN DEL MODELO
# ===========================
MODEL_NAME = "gpt2"  # Modelo ligero para pruebas rápidas
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()  # Modo inferencia (sin entrenamiento)

# Modelo de embeddings semánticos
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Modelo de análisis de sentimiento
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except Exception as e:
    print(f"[Aviso] No se pudo cargar sentiment analyzer: {e}")
    sentiment_analyzer = None
# ===========================
# MÉTRICA DE NOVEDAD SEMÁNTICA
# ===========================
def calcular_novedad_semantica(contexto, nuevo_texto):
    """
    Calcula la novedad semántica como 1 - similitud coseno entre embeddings del contexto y del nuevo texto.
    """
    if not contexto.strip() or not nuevo_texto.strip():
        return 0.0
    emb_contexto = embedder.encode([contexto])[0]
    emb_nuevo = embedder.encode([nuevo_texto])[0]
    sim = cosine_similarity([emb_contexto], [emb_nuevo])[0][0]
    return 1.0 - sim

# ===========================
# FUNCIONES CLAVE
# ===========================
def calcular_perplejidad(contexto, nuevo_texto):
    """
    Mide cuán "sorprendente" es el nuevo_texto dado el contexto (E_n).
    Devuelve la perplejidad promedio por token, normalizada por longitud.
    """
    # Tokenize context and new text separately to get accurate indices
    context_tokens = tokenizer(contexto, return_tensors="pt", truncation=True, max_length=1024)
    contexto_len = context_tokens["input_ids"].shape[1]
    
    # Tokenize full text
    texto_completo = contexto + " " + nuevo_texto if contexto.strip() else nuevo_texto
    inputs = tokenizer(texto_completo, return_tensors="pt", truncation=True, max_length=1024)
    
    # Extract new text token IDs (accounting for the space we added)
    full_len = inputs["input_ids"].shape[1]
    if contexto.strip():
        # Skip context tokens and the space token(s) that follow
        new_text_ids = inputs["input_ids"][0, contexto_len:]
    else:
        # No context, use all tokens from the new text
        new_text_ids = inputs["input_ids"][0, :]
    
    if len(new_text_ids) == 0:
        return 1.0
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits corresponding to the new text tokens
        # logits[i] predicts token at position i+1
        if contexto.strip():
            logits = outputs.logits[:, contexto_len-1:full_len-1, :]
        else:
            logits = outputs.logits[:, 0:full_len-1, :]
    
    log_probs = torch.log_softmax(logits, dim=-1)
    # Ensure shapes match for gather
    if logits.shape[1] != len(new_text_ids):
        # Fallback: just use the last tokens
        logits = outputs.logits[:, -len(new_text_ids):, :]
        log_probs = torch.log_softmax(logits, dim=-1)
    
    # Reshape index tensor to match log_probs shape
    index = new_text_ids.unsqueeze(0).unsqueeze(-1)
    token_log_probs = log_probs.gather(2, index).squeeze(-1)
    cross_entropy = -token_log_probs.mean().item()
    perplexity = np.exp(cross_entropy)
    return perplexity

def calcular_A(nuevo_texto, historial):
    """
    Calcula el Factor de Interacción (A) como función continua del ratio de tokens nuevos.
    """
    tokens_nuevos = set(tokenizer.encode(nuevo_texto))
    tokens_previos = set()
    for msg in historial[:-1]:
        tokens_previos.update(tokenizer.encode(msg))
    ratio_novedad = len(tokens_nuevos - tokens_previos) / len(tokens_nuevos) if tokens_nuevos else 0
    # A = 4*ratio_novedad - 2 (rango: -2 a 2)
    return 4 * ratio_novedad - 2

def calcular_D_KL(contexto, nuevo_texto):
    """
    Calcula D_KL entre distribuciones de probabilidad del siguiente token.
    Clippea probabilidades para estabilidad numérica.
    """
    from scipy.stats import entropy
    eps = 1e-12
    
    # If context is empty, use a simple token as context (e.g., start token)
    if not contexto.strip():
        contexto_para_modelo = "inicio"
    else:
        contexto_para_modelo = contexto
    
    # 1. Predicción ANTES (solo contexto)
    inputs_prev = tokenizer(contexto_para_modelo, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs_prev = model(**inputs_prev)
        probs_prev = torch.softmax(outputs_prev.logits[0, -1, :], dim=-1).cpu().numpy()
    # 2. Predicción DESPUÉS (contexto + nuevo texto)
    inputs_new = tokenizer(contexto_para_modelo + " " + nuevo_texto, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs_new = model(**inputs_new)
        probs_new = torch.softmax(outputs_new.logits[0, -1, :], dim=-1).cpu().numpy()
    # Clip
    probs_prev = np.clip(probs_prev, eps, 1.0)
    probs_new = np.clip(probs_new, eps, 1.0)
    # Normalizar
    probs_prev /= probs_prev.sum()
    probs_new /= probs_new.sum()
    # D_KL (P_prev || P_new)
    return float(entropy(probs_prev, probs_new))

# ===========================
# SIMULACIÓN DE CONVERSACIÓN
# ===========================
print("="*50)
print("SIMULACIÓN DE PLASTICIDAD ADAPTATIVA")
print("Escribe 'salir' para terminar. ¡Observa cómo cambia C_n+1!")
print("="*50)

historial = []
C_n = 0.0  # Inicializar plasticidad
integral = 0.0  # Acumulador para ∫(R·ΔK)dt
lambda_decay = 0.15  # Decaimiento de memoria
R_0 = 1.0
metricas = []  # Guardar métricas por interacción
import time
tiempos = []

while True:
    # 1. Entrada del usuario
    user_input = input("\nTú: ").strip()
    if user_input.lower() in ["salir", "exit"]:
        break
    t_actual = time.time()
    tiempos.append(t_actual)
    delta_t = 0 if len(tiempos) < 2 else tiempos[-1] - tiempos[-2]
    # 2. Actualizar historial
    historial.append(user_input)
    contexto = " ".join(historial[:-1]) if len(historial) > 1 else ""
    # 3. Calcular métricas clave
    A = calcular_A(user_input, historial)
    E_n = calcular_perplejidad(contexto, user_input)  # Energía de error predictivo
    D_KL = calcular_D_KL(contexto, user_input)        # Novedad (ΔK)
    novedad_semantica = calcular_novedad_semantica(contexto, user_input)  # Novedad semántica
    # 4. Actualizar integral evolutiva con decaimiento
    R_t = R_0 * np.exp(-lambda_decay * delta_t)
    integral = integral * np.exp(-lambda_decay * delta_t) + R_t * D_KL
    # 5. Calcular plasticidad actual (C_n+1)
    # Usar clipping para evitar overflow en exp
    modulador = 1.0 / (1.0 + np.exp(-np.clip(A * E_n, -50, 50)))  # σ(A·E_n) estable
    C_n = modulador * integral
    # 6. Calcular sentimiento del texto de entrada
    emo_score = 0.0
    if sentiment_analyzer is not None:
        try:
            sentiment_result = sentiment_analyzer(user_input[:512])[0]  # Limitar longitud
            emo_score = sentiment_result["score"] if sentiment_result["label"] == "POSITIVE" else 1.0 - sentiment_result["score"]
        except Exception as e:
            emo_score = 0.5  # Neutral por defecto si hay error
    
    # 7. Guardar métricas
    metricas.append({
        'A': A,
        'E_n': E_n,
        'D_KL': D_KL,
        'novedad_semantica': novedad_semantica,
        'R_t': R_t,
        'modulador': modulador,
        'C_n+1': C_n,
        'emo_score': emo_score,
        't': t_actual
    })
    # 8. Mostrar resultados
    print(f"\n[ANALISIS DE INTERACCION]")
    print(f"• Factor A: {A:.2f} (ratio_novedad continua)")
    print(f"• Novedad (D_KL): {D_KL:.4f}")
    print(f"• Novedad semantica: {novedad_semantica:.4f}")
    print(f"• Energia de error (E_n): {E_n:.4f}")
    print(f"• Memoria activa (R_t): {R_t:.4f}")
    print(f"• Sentimiento: {emo_score:.4f}")
    status = "ESTANCAMIENTO" if C_n < 0.5 else "ADAPTACION ACTIVA"
    print(f"• Plasticidad (C_n+1): {C_n:.4f} [{status}]")
    
    # Mensaje reflexivo cuando la plasticidad es baja
    if C_n < 0.5:
        print("\n>> El silencio se vuelve un espejo roto. Que necesitas para romper el ciclo?")
    # 9. Generar respuesta del chatbot (simulada)
    if A >= 0:
        respuesta = "¡Interesante! ¿Quieres profundizar en esto?"
    else:
        respuesta = "No entiendo tu punto. Repite con más claridad."
    historial.append(respuesta)
    print(f"IA: {respuesta}")

# ===========================
# RESUMEN FINAL
# ===========================
print("\n" + "="*50)
print("SIMULACIÓN FINALIZADA")
print(f"Plasticidad acumulada (C_final): {C_n:.4f}")
print(f"Total de interacciones: {len(historial)//2}")
print("="*50)
# Guardar métricas en archivo (opcional)
import json
with open("metricas_interaccion.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)
