import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import time
import os

# ═══════════════════════════════════════════════════════
#  COLORES ANSI
# ═══════════════════════════════════════════════════════
R   = "\033[0m"
CY  = "\033[96m"    # cyan       — títulos / menú
MG  = "\033[95m"    # magenta    — usuario
GR  = "\033[92m"    # verde      — IA / OK
YL  = "\033[93m"    # amarillo   — métricas
DM  = "\033[2m"     # dim        — info secundaria
RD  = "\033[91m"    # rojo       — alertas
BL  = "\033[94m"    # azul       — plasticidad
WH  = "\033[97m"    # blanco     — valores

# ═══════════════════════════════════════════════════════
#  MODELOS SUGERIDOS (con descripción)
# ═══════════════════════════════════════════════════════
MODELOS_SUGERIDOS = [
    ("mistral",         "Mistral 7B — equilibrado, buena coherencia"),
    ("llama3.1:8b",     "Llama 3.1 8B — razonamiento sólido"),
    ("gemma3:4b",       "Gemma3 4B — liviano y rápido"),
    ("qwen3:8b",        "Qwen3 8B — multilingüe avanzado"),
    ("deepseek-r1:8b",  "DeepSeek R1 — razonamiento profundo"),
    ("dolphin3",        "Dolphin3 — sin restricciones"),
    ("qwen3-vl:4b",     "Qwen3-VL 4B — visión + lenguaje"),
]

OLLAMA_URL  = "http://localhost:11434"
OLLAMA_CHAT = f"{OLLAMA_URL}/api/chat"
OLLAMA_TAGS = f"{OLLAMA_URL}/api/tags"

# ═══════════════════════════════════════════════════════
#  DETECTAR MODELOS INSTALADOS EN OLLAMA
# ═══════════════════════════════════════════════════════
def obtener_modelos_ollama():
    """Consulta a Ollama qué modelos están instalados localmente."""
    try:
        r = requests.get(OLLAMA_TAGS, timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.exceptions.ConnectionError:
        return None   # Ollama no está corriendo
    except Exception:
        return []

# ═══════════════════════════════════════════════════════
#  MENÚ DE SELECCIÓN DE MODELO
# ═══════════════════════════════════════════════════════
def menu_seleccion_modelo():
    print("\n" + "═"*58)
    print(f"{CY}   🧠  PLASTICIDAD ADAPTATIVA — Selección de Modelo{R}")
    print("═"*58)

    # Consultar Ollama
    instalados = obtener_modelos_ollama()

    if instalados is None:
        print(f"{RD}  ⚠  Ollama no responde en {OLLAMA_URL}{R}")
        print(f"{DM}     Asegúrate de que Ollama esté corriendo (`ollama serve`){R}")
        print(f"{DM}     Puedes igualmente ingresar un nombre y probar.{R}\n")
        instalados = []

    # Cruzar sugeridos con instalados
    print(f"\n{CY}  MODELOS SUGERIDOS{R}  {DM}(✓ = instalado en tu Ollama){R}\n")
    opciones_visibles = []

    for nombre, desc in MODELOS_SUGERIDOS:
        # Comparación flexible (sin tag :latest)
        base = nombre.split(":")[0]
        esta = any(base in m for m in instalados)
        marca = f"{GR}✓{R}" if esta else f"{DM}·{R}"
        idx = len(opciones_visibles) + 1
        opciones_visibles.append(nombre)
        print(f"  {CY}[{idx}]{R} {marca} {WH}{nombre:<22}{R} {DM}{desc}{R}")

    # Modelos instalados que no están en la lista sugerida
    extras = []
    for m in instalados:
        base_m = m.split(":")[0]
        if not any(base_m in s[0] for s in MODELOS_SUGERIDOS):
            extras.append(m)

    if extras:
        print(f"\n{CY}  OTROS MODELOS INSTALADOS{R}\n")
        for m in extras:
            idx = len(opciones_visibles) + 1
            opciones_visibles.append(m)
            print(f"  {CY}[{idx}]{R} {GR}✓{R} {WH}{m}{R}")

    # Opción manual
    idx_manual = len(opciones_visibles) + 1
    print(f"\n  {CY}[{idx_manual}]{R} {YL}✎  Ingresar nombre manualmente{R}")
    print(f"  {CY}[0]{R}  Salir")
    print("═"*58)

    while True:
        try:
            eleccion = input(f"\n{CY}  Elige una opción: {R}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if eleccion == "0":
            return None

        if eleccion == str(idx_manual):
            modelo = _ingresar_modelo_manual(instalados)
            if modelo:
                return modelo
            continue

        if eleccion.isdigit():
            idx = int(eleccion) - 1
            if 0 <= idx < len(opciones_visibles):
                modelo = opciones_visibles[idx]
                # Verificar si está instalado
                base = modelo.split(":")[0]
                esta = any(base in m for m in instalados)
                if not esta and instalados is not None:
                    print(f"{YL}  ⚠  '{modelo}' no parece estar instalado.{R}")
                    confirmar = input(f"{DM}     ¿Intentar de todas formas? (s/n): {R}").strip().lower()
                    if confirmar not in ("s", "si", "sí", "y", "yes"):
                        continue
                return modelo

        print(f"{RD}  Opción no válida.{R}")


def _ingresar_modelo_manual(instalados):
    print(f"\n{CY}  Ingresa el nombre exacto del modelo Ollama{R}")
    print(f"  {DM}Ejemplos: mistral, llama3:8b, phi3:mini, gemma:2b{R}")
    if instalados:
        print(f"  {DM}Instalados detectados: {', '.join(instalados[:6])}{'...' if len(instalados)>6 else ''}{R}")
    try:
        nombre = input(f"\n{CY}  Nombre del modelo: {R}").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not nombre:
        print(f"{RD}  Nombre vacío — cancelado.{R}")
        return None
    print(f"{GR}  ✓ Modelo seleccionado: {nombre}{R}")
    return nombre


# ═══════════════════════════════════════════════════════
#  GENERACIÓN DE RESPUESTA VÍA OLLAMA
# ═══════════════════════════════════════════════════════
def generar_respuesta_ollama(modelo_ollama, contexto, user_input, historial_chat):
    """Genera respuesta usando Ollama con historial de conversación."""
    messages = []
    # Agregar historial como pares user/assistant
    for i in range(0, len(historial_chat) - 1, 2):
        if i < len(historial_chat):
            messages.append({"role": "user",      "content": historial_chat[i]})
        if i + 1 < len(historial_chat):
            messages.append({"role": "assistant", "content": historial_chat[i+1]})
    # Mensaje actual
    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": modelo_ollama,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.85, "top_p": 0.9}
    }
    try:
        r = requests.post(OLLAMA_CHAT, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip(), None
    except requests.exceptions.ConnectionError:
        return None, "Ollama no responde. ¿Está corriendo?"
    except requests.exceptions.Timeout:
        return None, "Tiempo de espera agotado."
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════
#  CONFIGURACIÓN GPT2 (métricas de plasticidad)
# ═══════════════════════════════════════════════════════
print(f"\n{DM}[Cargando modelos de análisis (GPT2 + embeddings)...]{R}")

GPT2_NAME = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(GPT2_NAME)
gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_NAME)
gpt2_model.eval()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except Exception as e:
    print(f"{YL}[Aviso] No se pudo cargar sentiment analyzer: {e}{R}")
    sentiment_analyzer = None

print(f"{GR}[✓ Modelos de análisis listos]{R}")


# ═══════════════════════════════════════════════════════
#  MÉTRICAS DE PLASTICIDAD (sin cambios de tu original)
# ═══════════════════════════════════════════════════════
def calcular_novedad_semantica(contexto, nuevo_texto):
    if not contexto.strip() or not nuevo_texto.strip():
        return 0.0
    emb_c = embedder.encode([contexto])[0]
    emb_n = embedder.encode([nuevo_texto])[0]
    sim = cosine_similarity([emb_c], [emb_n])[0][0]
    return 1.0 - sim


def calcular_perplejidad(contexto, nuevo_texto):
    context_tokens = tokenizer(contexto, return_tensors="pt", truncation=True, max_length=1024)
    contexto_len = context_tokens["input_ids"].shape[1]
    texto_completo = contexto + " " + nuevo_texto if contexto.strip() else nuevo_texto
    inputs = tokenizer(texto_completo, return_tensors="pt", truncation=True, max_length=1024)
    full_len = inputs["input_ids"].shape[1]
    if contexto.strip():
        new_text_ids = inputs["input_ids"][0, contexto_len:]
    else:
        new_text_ids = inputs["input_ids"][0, :]
    if len(new_text_ids) == 0:
        return 1.0
    with torch.no_grad():
        outputs = gpt2_model(**inputs)
        if contexto.strip():
            logits = outputs.logits[:, contexto_len-1:full_len-1, :]
        else:
            logits = outputs.logits[:, 0:full_len-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    if logits.shape[1] != len(new_text_ids):
        logits = outputs.logits[:, -len(new_text_ids):, :]
        log_probs = torch.log_softmax(logits, dim=-1)
    index = new_text_ids.unsqueeze(0).unsqueeze(-1)
    token_log_probs = log_probs.gather(2, index).squeeze(-1)
    cross_entropy = -token_log_probs.mean().item()
    return np.exp(cross_entropy)


def calcular_A(nuevo_texto, historial):
    tokens_nuevos = set(tokenizer.encode(nuevo_texto))
    tokens_previos = set()
    for msg in historial[:-1]:
        tokens_previos.update(tokenizer.encode(msg))
    ratio = len(tokens_nuevos - tokens_previos) / len(tokens_nuevos) if tokens_nuevos else 0
    return 4 * ratio - 2


def calcular_D_KL(contexto, nuevo_texto):
    from scipy.stats import entropy
    eps = 1e-12
    ctx = contexto if contexto.strip() else "inicio"
    inputs_prev = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        p_prev = torch.softmax(gpt2_model(**inputs_prev).logits[0, -1, :], dim=-1).cpu().numpy()
    inputs_new = tokenizer(ctx + " " + nuevo_texto, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        p_new = torch.softmax(gpt2_model(**inputs_new).logits[0, -1, :], dim=-1).cpu().numpy()
    p_prev = np.clip(p_prev, eps, 1.0); p_prev /= p_prev.sum()
    p_new  = np.clip(p_new,  eps, 1.0); p_new  /= p_new.sum()
    return float(entropy(p_prev, p_new))


# ═══════════════════════════════════════════════════════
#  BUCLE PRINCIPAL
# ═══════════════════════════════════════════════════════
def run_sesion(modelo_ollama):
    print("\n" + "═"*58)
    print(f"{CY}  🧬 PLASTICIDAD ADAPTATIVA  ·  Modelo: {GR}{modelo_ollama}{R}")
    print(f"  {DM}Escribe 'salir' o '/chao' para volver al menú.{R}")
    print("═"*58)

    historial   = []
    C_n         = 0.0
    integral    = 0.0
    lambda_d    = 0.15
    R_0         = 1.0
    metricas    = []
    tiempos     = []

    while True:
        # ── Entrada ──────────────────────────────────────────
        try:
            user_input = input(f"\n{MG}Tú:{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ["salir", "exit", "/chao", "/menu"]:
            break
        if not user_input:
            continue

        t_actual = time.time()
        tiempos.append(t_actual)
        delta_t = 0 if len(tiempos) < 2 else tiempos[-1] - tiempos[-2]

        historial.append(user_input)
        contexto = " ".join(historial[:-1]) if len(historial) > 1 else ""

        # ── Métricas ──────────────────────────────────────────
        print(f"{DM}  [calculando métricas...]{R}", end="\r")
        A                = calcular_A(user_input, historial)
        E_n              = calcular_perplejidad(contexto, user_input)
        D_KL             = calcular_D_KL(contexto, user_input)
        novedad_sem      = calcular_novedad_semantica(contexto, user_input)

        R_t      = R_0 * np.exp(-lambda_d * delta_t)
        integral = integral * np.exp(-lambda_d * delta_t) + R_t * D_KL
        modulador = 1.0 / (1.0 + np.exp(-np.clip(A * E_n, -50, 50)))
        C_n      = modulador * integral

        emo_score = 0.5
        if sentiment_analyzer:
            try:
                res = sentiment_analyzer(user_input[:512])[0]
                emo_score = res["score"] if res["label"] == "POSITIVE" else 1.0 - res["score"]
            except Exception:
                pass

        metricas.append({
            "A": A, "E_n": E_n, "D_KL": D_KL,
            "novedad_semantica": novedad_sem,
            "R_t": R_t, "modulador": modulador,
            "C_n+1": C_n, "emo_score": emo_score, "t": t_actual
        })

        # ── Mostrar métricas ──────────────────────────────────
        status = f"{RD}ESTANCAMIENTO{R}" if C_n < 0.5 else f"{GR}ADAPTACIÓN ACTIVA{R}"
        bar_len  = 30
        bar_fill = int(min(C_n, 5.0) / 5.0 * bar_len)
        bar      = f"{GR}{'█' * bar_fill}{DM}{'░' * (bar_len - bar_fill)}{R}"

        print(f"\n{YL}╔══ ANÁLISIS DE INTERACCIÓN ══════════════════════╗{R}")
        print(f"{YL}║{R}  Factor A          {WH}{A:+.4f}{R}  {DM}(ratio novedad léxica){R}")
        print(f"{YL}║{R}  Novedad D_KL      {WH}{D_KL:.4f}{R}")
        print(f"{YL}║{R}  Novedad semántica {WH}{novedad_sem:.4f}{R}")
        print(f"{YL}║{R}  Energía E_n       {WH}{E_n:.4f}{R}  {DM}(perplejidad){R}")
        print(f"{YL}║{R}  Memoria R_t       {WH}{R_t:.4f}{R}")
        print(f"{YL}║{R}  Sentimiento       {WH}{emo_score:.4f}{R}")
        print(f"{YL}║{R}  Modulador σ(A·E)  {WH}{modulador:.4f}{R}")
        print(f"{YL}╠══ PLASTICIDAD C_n+1 ═══════════════════════════╣{R}")
        print(f"{YL}║{R}  {bar}  {BL}{C_n:.4f}{R}")
        print(f"{YL}║{R}  Estado: {status}")
        print(f"{YL}╚════════════════════════════════════════════════╝{R}")

        if C_n < 0.5:
            print(f"\n{DM}  >> El silencio se vuelve un espejo roto. ¿Qué necesitas para romper el ciclo?{R}")

        # ── Respuesta Ollama ──────────────────────────────────
        print(f"{DM}  [generando respuesta con {modelo_ollama}...]{R}", end="\r")
        respuesta, error = generar_respuesta_ollama(modelo_ollama, contexto, user_input, historial)

        if error:
            print(f"{RD}  ⚠ Error Ollama: {error}{R}")
            # Fallback simple
            respuesta = "¡Interesante! ¿Quieres profundizar en esto?" if A >= 0 else "No entiendo tu punto. Repite con más claridad."

        historial.append(respuesta)
        print(f"\n{GR}{modelo_ollama}:{R} {respuesta}\n")

    # ── Resumen final ─────────────────────────────────────
    print("\n" + "═"*58)
    print(f"{CY}  SESIÓN FINALIZADA{R}")
    print(f"  Plasticidad final (C_n): {BL}{C_n:.4f}{R}")
    print(f"  Total interacciones:     {WH}{len(historial)//2}{R}")
    print("═"*58)

    if metricas:
        fname = f"metricas_{modelo_ollama.replace(':', '_').replace('.', '_')}.json"
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(metricas, f, ensure_ascii=False, indent=2)
            print(f"{DM}  Métricas guardadas en: {fname}{R}")
        except Exception as e:
            print(f"{RD}  No se pudieron guardar métricas: {e}{R}")


# ═══════════════════════════════════════════════════════
#  ENTRADA PRINCIPAL
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{CY}  🧬 PLASTICIDAD ADAPTATIVA v2.0{R}")
    print(f"  {DM}Análisis neuroadaptativo con modelos Ollama locales{R}")

    while True:
        modelo = menu_seleccion_modelo()

        if modelo is None:
            print(f"\n{GR}  👋 ¡Hasta pronto, Germán el Gris!{R}\n")
            break

        run_sesion(modelo)

        print(f"\n{DM}  [Volviendo al menú de selección...]{R}")
