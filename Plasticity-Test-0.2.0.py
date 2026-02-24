import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import time
import os
from datetime import datetime

# ═══════════════════════════════════════════════════════
#  COLORES ANSI
# ═══════════════════════════════════════════════════════
R   = "\033[0m"
CY  = "\033[96m"
MG  = "\033[95m"
GR  = "\033[92m"
YL  = "\033[93m"
DM  = "\033[2m"
RD  = "\033[91m"
BL  = "\033[94m"
WH  = "\033[97m"

# ═══════════════════════════════════════════════════════
#  MODELOS SUGERIDOS
# ═══════════════════════════════════════════════════════
MODELOS_SUGERIDOS = [
    ("mistral",         "Mistral 7B — equilibrado, buena coherencia"),
    ("llama3.1:8b",     "Llama 3.1 8B — razonamiento solido"),
    ("gemma3:4b",       "Gemma3 4B — liviano y rapido"),
    ("qwen3:8b",        "Qwen3 8B — multilingue avanzado"),
    ("deepseek-r1:8b",  "DeepSeek R1 — razonamiento profundo"),
    ("dolphin3",        "Dolphin3 — sin restricciones"),
    ("qwen3-vl:4b",     "Qwen3-VL 4B — vision + lenguaje"),
]

OLLAMA_URL  = "http://localhost:11434"
OLLAMA_CHAT = f"{OLLAMA_URL}/api/chat"
OLLAMA_TAGS = f"{OLLAMA_URL}/api/tags"


# ═══════════════════════════════════════════════════════
#  DETECTAR MODELOS EN OLLAMA
# ═══════════════════════════════════════════════════════
def obtener_modelos_ollama():
    try:
        r = requests.get(OLLAMA_TAGS, timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return []


# ═══════════════════════════════════════════════════════
#  MENU DE SELECCION DE MODELO
# ═══════════════════════════════════════════════════════
def menu_seleccion_modelo():
    print("\n" + "="*58)
    print(f"{CY}   PLASTICIDAD ADAPTATIVA v3.0 -- Seleccion de Modelo{R}")
    print("="*58)

    instalados = obtener_modelos_ollama()

    if instalados is None:
        print(f"{RD}  Ollama no responde en {OLLAMA_URL}{R}")
        print(f"{DM}  Ejecuta: ollama serve{R}\n")
        instalados = []

    print(f"\n{CY}  MODELOS SUGERIDOS{R}  {DM}(check = instalado){R}\n")
    opciones_visibles = []

    for nombre, desc in MODELOS_SUGERIDOS:
        base  = nombre.split(":")[0]
        esta  = any(base in m for m in instalados)
        marca = f"{GR}[OK]{R}" if esta else f"{DM}[ ]{R}"
        idx   = len(opciones_visibles) + 1
        opciones_visibles.append(nombre)
        print(f"  {CY}[{idx}]{R} {marca} {WH}{nombre:<22}{R} {DM}{desc}{R}")

    extras = [m for m in instalados
              if not any(m.split(":")[0] in s[0] for s in MODELOS_SUGERIDOS)]
    if extras:
        print(f"\n{CY}  OTROS MODELOS INSTALADOS{R}\n")
        for m in extras:
            idx = len(opciones_visibles) + 1
            opciones_visibles.append(m)
            print(f"  {CY}[{idx}]{R} {GR}[OK]{R} {WH}{m}{R}")

    idx_manual = len(opciones_visibles) + 1
    print(f"\n  {CY}[{idx_manual}]{R} {YL}Ingresar nombre manualmente{R}")
    print(f"  {CY}[0]{R}  Salir")
    print("="*58)

    while True:
        try:
            eleccion = input(f"\n{CY}  Elige una opcion: {R}").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return None

        if eleccion == "0":
            return None
        if eleccion == str(idx_manual):
            modelo = _ingresar_modelo_manual(instalados)
            if modelo: return modelo
            continue
        if eleccion.isdigit():
            idx = int(eleccion) - 1
            if 0 <= idx < len(opciones_visibles):
                modelo = opciones_visibles[idx]
                base   = modelo.split(":")[0]
                esta   = any(base in m for m in instalados)
                if not esta and instalados is not None:
                    print(f"{YL}  '{modelo}' no parece instalado.{R}")
                    conf = input(f"{DM}  Intentar igual? (s/n): {R}").strip().lower()
                    if conf not in ("s", "si", "y", "yes"): continue
                return modelo
        print(f"{RD}  Opcion no valida.{R}")


def _ingresar_modelo_manual(instalados):
    print(f"\n{CY}  Ingresa el nombre exacto del modelo{R}")
    if instalados:
        print(f"  {DM}Instalados: {', '.join(instalados[:6])}{'...' if len(instalados)>6 else ''}{R}")
    try:
        nombre = input(f"\n{CY}  Nombre: {R}").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not nombre:
        print(f"{RD}  Nombre vacio.{R}"); return None
    print(f"{GR}  Seleccionado: {nombre}{R}")
    return nombre


# ═══════════════════════════════════════════════════════
#  GENERACION VIA OLLAMA
# ═══════════════════════════════════════════════════════
def generar_respuesta_ollama(modelo_ollama, historial_completo):
    messages = []
    for i in range(0, len(historial_completo), 2):
        messages.append({"role": "user", "content": historial_completo[i]})
        if i + 1 < len(historial_completo):
            messages.append({"role": "assistant", "content": historial_completo[i+1]})

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
        return None, "Ollama no responde."
    except requests.exceptions.Timeout:
        return None, "Tiempo agotado."
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════
#  CARGAR MODELOS DE ANALISIS
# ═══════════════════════════════════════════════════════
print(f"\n{DM}[Cargando GPT2 + embeddings para analisis...]{R}")

tokenizer  = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except Exception as e:
    print(f"{YL}[Aviso] Sentiment analyzer no disponible: {e}{R}")
    sentiment_analyzer = None

print(f"{GR}[Modelos de analisis listos]{R}")


# ═══════════════════════════════════════════════════════
#  METRICAS -- calculadas sobre la RESPUESTA del modelo
# ═══════════════════════════════════════════════════════
def calcular_novedad_semantica(contexto, respuesta):
    if not contexto.strip() or not respuesta.strip():
        return 0.0
    emb_c = embedder.encode([contexto])[0]
    emb_r = embedder.encode([respuesta])[0]
    return float(1.0 - cosine_similarity([emb_c], [emb_r])[0][0])


def calcular_perplejidad(contexto, respuesta):
    ctx_tok   = tokenizer(contexto, return_tensors="pt", truncation=True, max_length=1024)
    ctx_len   = ctx_tok["input_ids"].shape[1]
    full_txt  = (contexto + " " + respuesta) if contexto.strip() else respuesta
    inputs    = tokenizer(full_txt, return_tensors="pt", truncation=True, max_length=1024)
    full_len  = inputs["input_ids"].shape[1]

    new_ids = inputs["input_ids"][0, ctx_len:] if contexto.strip() else inputs["input_ids"][0, :]
    if len(new_ids) == 0:
        return 1.0

    with torch.no_grad():
        out = gpt2_model(**inputs)
        logits = out.logits[:, ctx_len-1:full_len-1, :] if contexto.strip() else out.logits[:, 0:full_len-1, :]

    log_p = torch.log_softmax(logits, dim=-1)
    if logits.shape[1] != len(new_ids):
        logits = out.logits[:, -len(new_ids):, :]
        log_p  = torch.log_softmax(logits, dim=-1)

    idx    = new_ids.unsqueeze(0).unsqueeze(-1)
    tlp    = log_p.gather(2, idx).squeeze(-1)
    return float(np.exp(-tlp.mean().item()))


def calcular_A(respuesta, respuestas_previas):
    tokens_n = set(tokenizer.encode(respuesta))
    tokens_p = set()
    for r in respuestas_previas:
        tokens_p.update(tokenizer.encode(r))
    ratio = len(tokens_n - tokens_p) / len(tokens_n) if tokens_n else 0
    return float(4 * ratio - 2)


def calcular_D_KL(contexto, respuesta):
    from scipy.stats import entropy
    eps = 1e-12
    ctx = contexto if contexto.strip() else "inicio"

    inp_p = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        p_p = torch.softmax(gpt2_model(**inp_p).logits[0, -1, :], dim=-1).cpu().numpy()

    inp_n = tokenizer(ctx + " " + respuesta, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        p_n = torch.softmax(gpt2_model(**inp_n).logits[0, -1, :], dim=-1).cpu().numpy()

    p_p = np.clip(p_p, eps, 1.0); p_p /= p_p.sum()
    p_n = np.clip(p_n, eps, 1.0); p_n /= p_n.sum()
    return float(entropy(p_p, p_n))


def calcular_longitud_norm(respuesta, max_chars=2000):
    return min(len(respuesta) / max_chars, 1.0)


# ═══════════════════════════════════════════════════════
#  INFORME FINAL CON GRAFICOS
# ═══════════════════════════════════════════════════════
def generar_informe(modelo_ollama, metricas, timestamp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print(f"{RD}  matplotlib no instalado. Ejecuta: pip install matplotlib{R}")
        return None

    if not metricas:
        return None

    # Series
    n           = list(range(1, len(metricas) + 1))
    plasticidad = [m["C_n+1"]             for m in metricas]
    energia     = [min(m["E_n"], 300)     for m in metricas]
    dkl         = [m["D_KL"]              for m in metricas]
    novedad_s   = [m["novedad_semantica"] for m in metricas]
    factor_a    = [m["A"]                 for m in metricas]
    sentimiento = [m["emo_score"]         for m in metricas]
    longitud    = [m["longitud_norm"]     for m in metricas]

    # Valores radar normalizados a [0,1]
    def norm01(val, vmin, vmax):
        return max(0.0, min(1.0, (val - vmin) / (vmax - vmin + 1e-9)))

    radar_labels = [
        "Plasticidad", "Creatividad\n(E_n)", "Impacto\n(D_KL)",
        "Novedad\nSemantica", "Vocabulario\nNuevo (A)",
        "Sentimiento+", "Longitud"
    ]
    radar_vals = [
        norm01(np.mean(plasticidad), 0,  5),
        norm01(np.mean(energia),     1,  200),
        norm01(np.mean(dkl),         0,  5),
        np.mean(novedad_s),
        norm01(np.mean(factor_a),   -2,  2),
        np.mean(sentimiento),
        np.mean(longitud),
    ]

    # Estilo oscuro
    DARK    = "#0d0f14"
    MID     = "#13161e"
    BORDER  = "#252a3d"
    TXT     = "#e8eaf6"
    TXDIM   = "#666e8a"
    CYAN_C  = "#00e5ff"
    PURP    = "#7c4dff"
    GREEN_C = "#00e676"
    YEL     = "#ffab40"
    PINK_C  = "#ff6b9d"
    RED_C   = "#ff5252"

    plt.rcParams.update({
        "figure.facecolor": DARK, "axes.facecolor": MID,
        "axes.edgecolor": BORDER, "axes.labelcolor": TXT,
        "xtick.color": TXDIM, "ytick.color": TXDIM,
        "text.color": TXT, "grid.color": BORDER,
        "grid.linestyle": "--", "grid.alpha": 0.5,
        "font.family": "monospace", "axes.titlesize": 11,
        "axes.titlecolor": CYAN_C,
    })

    fig = plt.figure(figsize=(18, 14), facecolor=DARK)
    fig.suptitle(
        f"INFORME DE PLASTICIDAD ADAPTATIVA  |  Modelo: {modelo_ollama}\n"
        f"{timestamp}  |  {len(metricas)} interacciones",
        fontsize=14, color=CYAN_C, y=0.98, fontfamily="monospace"
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4,
                  left=0.06, right=0.97, top=0.91, bottom=0.06)

    # ── 1. RADAR ─────────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[0, 0], projection="polar")
    ax_r.set_facecolor(MID)
    num_v   = len(radar_labels)
    angles  = np.linspace(0, 2*np.pi, num_v, endpoint=False).tolist()
    vals_cl = radar_vals + [radar_vals[0]]
    ang_cl  = angles + [angles[0]]

    for lvl in [0.25, 0.5, 0.75, 1.0]:
        ax_r.plot(ang_cl, [lvl]*(num_v+1), color=BORDER, lw=0.8)
    ax_r.fill(angles, radar_vals, alpha=0.25, color=CYAN_C)
    ax_r.plot(ang_cl, vals_cl, color=CYAN_C, lw=2.5)
    ax_r.scatter(angles, radar_vals, color=CYAN_C, s=60, edgecolors="white", lw=0.8, zorder=5)
    ax_r.set_xticks(angles)
    ax_r.set_xticklabels(radar_labels, fontsize=8.5, color=TXT)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color=TXDIM)
    ax_r.set_ylim(0, 1)
    ax_r.spines["polar"].set_color(BORDER)
    ax_r.set_title("Perfil Global del Modelo", pad=18)
    # Promedio plasticidad como subtitulo
    ax_r.text(0, -0.3, f"C media = {np.mean(plasticidad):.3f}",
              ha="center", fontsize=9, color=CYAN_C, transform=ax_r.transData)

    # ── 2. PLASTICIDAD C_n+1 ──────────────────────────────────────────────────
    ax_p = fig.add_subplot(gs[0, 1])
    ax_p.plot(n, plasticidad, color=CYAN_C, lw=2, zorder=3, label="C_n+1")
    ax_p.fill_between(n, plasticidad, alpha=0.15, color=CYAN_C)
    ax_p.axhline(0.5, color=RED_C, lw=1, ls="--", alpha=0.7, label="Umbral (0.5)")
    ax_p.axhline(np.mean(plasticidad), color=YEL, lw=1, ls=":",
                 label=f"Media ({np.mean(plasticidad):.3f})")
    ax_p.axhspan(0, 0.5, alpha=0.05, color=RED_C)
    ax_p.axhspan(0.5, max(plasticidad + [1.0])*1.1, alpha=0.05, color=GREEN_C)

    idx_max = int(np.argmax(plasticidad))
    idx_min = int(np.argmin(plasticidad))
    offset  = max(plasticidad) * 0.12
    ax_p.annotate(f"MAX\n{plasticidad[idx_max]:.3f}",
                  xy=(n[idx_max], plasticidad[idx_max]),
                  xytext=(n[idx_max], plasticidad[idx_max] + offset),
                  fontsize=7, color=GREEN_C, ha="center",
                  arrowprops=dict(arrowstyle="->", color=GREEN_C, lw=1.2))
    ax_p.annotate(f"MIN\n{plasticidad[idx_min]:.3f}",
                  xy=(n[idx_min], plasticidad[idx_min]),
                  xytext=(n[idx_min], plasticidad[idx_min] - offset),
                  fontsize=7, color=RED_C, ha="center",
                  arrowprops=dict(arrowstyle="->", color=RED_C, lw=1.2))

    ax_p.set_title("Plasticidad C_n+1 por Interaccion")
    ax_p.set_xlabel("Interaccion #", fontsize=9)
    ax_p.set_ylabel("C_n+1", fontsize=9)
    ax_p.legend(fontsize=8, facecolor=MID, edgecolor=BORDER, labelcolor=TXT)
    ax_p.grid(True); ax_p.set_xticks(n)

    # ── 3. MULTI-LINEA metricas ───────────────────────────────────────────────
    ax_m = fig.add_subplot(gs[0, 2])
    en_n  = [e / (max(energia)+1e-9)  for e in energia]
    dkl_n = [d / (max(dkl)+1e-9)     for d in dkl]
    ax_m.plot(n, en_n,      color=YEL,    lw=2, marker="o", ms=4, label="E_n (norm)")
    ax_m.plot(n, dkl_n,     color=PURP,   lw=2, marker="s", ms=4, label="D_KL (norm)")
    ax_m.plot(n, novedad_s, color=PINK_C, lw=2, marker="^", ms=4, label="Nov. Semantica")
    ax_m.plot(n, sentimiento, color=GREEN_C, lw=1.5, ls="--", alpha=0.8, label="Sentimiento")
    ax_m.set_title("Metricas de Respuesta (normalizadas)")
    ax_m.set_xlabel("Interaccion #", fontsize=9)
    ax_m.set_ylabel("Valor [0..1]", fontsize=9)
    ax_m.legend(fontsize=7.5, facecolor=MID, edgecolor=BORDER, labelcolor=TXT)
    ax_m.grid(True); ax_m.set_xticks(n); ax_m.set_ylim(-0.05, 1.1)

    # ── 4. BARRAS Factor A ────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[1, 0])
    bcolors = [GREEN_C if a >= 0 else RED_C for a in factor_a]
    bars = ax_a.bar(n, factor_a, color=bcolors, alpha=0.85, edgecolor=BORDER, lw=0.8)
    ax_a.axhline(0, color=TXT, lw=0.8, alpha=0.4)
    ax_a.axhline(np.mean(factor_a), color=YEL, lw=1.2, ls="--",
                 label=f"Media ({np.mean(factor_a):+.3f})")
    for bar, val in zip(bars, factor_a):
        ax_a.text(bar.get_x() + bar.get_width()/2,
                  val + (0.05 if val >= 0 else -0.12),
                  f"{val:+.2f}", ha="center", fontsize=7, color=TXT)
    p1 = mpatches.Patch(color=GREEN_C, label="Vocabulario nuevo (+)")
    p2 = mpatches.Patch(color=RED_C,   label="Vocabulario repetido (-)")
    ax_a.legend(handles=[p1, p2], fontsize=7.5, facecolor=MID, edgecolor=BORDER, labelcolor=TXT)
    ax_a.set_title("Factor A -- Novedad Lexica del Modelo")
    ax_a.set_xlabel("Interaccion #", fontsize=9)
    ax_a.set_ylabel("Factor A [-2 ... +2]", fontsize=9)
    ax_a.set_xticks(n); ax_a.set_ylim(-2.4, 2.4); ax_a.grid(True, axis="y")

    # ── 5. DISPERSION E_n vs D_KL, color = plasticidad ───────────────────────
    ax_sc = fig.add_subplot(gs[1, 1])
    sc = ax_sc.scatter(energia, dkl, c=plasticidad, cmap="plasma",
                       s=100, alpha=0.85, edgecolors="white", lw=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax_sc, pad=0.02)
    cbar.set_label("Plasticidad C_n+1", fontsize=8, color=TXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TXT, fontsize=7)
    for i, (e, d) in enumerate(zip(energia, dkl)):
        ax_sc.annotate(str(n[i]), (e, d), fontsize=7, color=TXT,
                       textcoords="offset points", xytext=(5, 3))
    ax_sc.set_title("Creatividad vs Impacto Informacional")
    ax_sc.set_xlabel("E_n -- Creatividad (perplejidad)", fontsize=9)
    ax_sc.set_ylabel("D_KL -- Impacto informacional", fontsize=9)
    ax_sc.grid(True)

    # ── 6. TABLA RESUMEN ──────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.set_facecolor(DARK); ax_t.axis("off")

    def fmt(arr):
        return f"{np.mean(arr):.4f}  (sigma={np.std(arr):.4f})"

    estado_final  = "ADAPTACION ACTIVA" if plasticidad[-1] >= 0.5 else "ESTANCAMIENTO"
    color_estado  = GREEN_C if plasticidad[-1] >= 0.5 else RED_C

    filas = [
        ("Modelo evaluado",       modelo_ollama),
        ("Interacciones",         str(len(metricas))),
        ("Fecha / Hora",          timestamp),
        ("---", "---"),
        ("Plasticidad C_n+1",     fmt(plasticidad)),
        ("Creatividad E_n",       fmt(energia)),
        ("Impacto D_KL",          fmt(dkl)),
        ("Novedad semantica",     fmt(novedad_s)),
        ("Factor A",              fmt(factor_a)),
        ("Sentimiento resp.",     fmt(sentimiento)),
        ("Longitud norm.",        fmt(longitud)),
        ("---", "---"),
        ("C_n+1 maxima",  f"{max(plasticidad):.4f}  (#{ np.argmax(plasticidad)+1})"),
        ("C_n+1 minima",  f"{min(plasticidad):.4f}  (#{ np.argmin(plasticidad)+1})"),
        ("Estado final",          estado_final),
    ]

    y = 0.97
    for etiq, valor in filas:
        if etiq == "---":
            ax_t.axhline(y + 0.005, color=BORDER, lw=0.8,
                         xmin=0.02, xmax=0.98, transform=ax_t.transAxes)
            y -= 0.03; continue
        col_v = color_estado if etiq == "Estado final" else TXT
        ax_t.text(0.03, y, etiq + ":", fontsize=8.5, color=TXDIM,
                  transform=ax_t.transAxes, va="top")
        ax_t.text(0.97, y, valor, fontsize=8.5, color=col_v,
                  transform=ax_t.transAxes, va="top", ha="right")
        y -= 0.062

    ax_t.set_title("Estadisticas de la Sesion")

    # ── Guardar ───────────────────────────────────────────────────────────────
    safe_m  = modelo_ollama.replace(":", "_").replace(".", "_")
    safe_ts = timestamp.replace(" ", "_").replace(":", "-")
    png_out  = f"informe_{safe_m}_{safe_ts}.png"
    json_out = f"metricas_{safe_m}_{safe_ts}.json"

    try:
        fig.savefig(png_out, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)
        print(f"{GR}  Informe guardado: {png_out}{R}")
    except Exception as e:
        print(f"{RD}  Error guardando PNG: {e}{R}")
        png_out = None

    try:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(metricas, f, ensure_ascii=False, indent=2)
        print(f"{GR}  Metricas JSON: {json_out}{R}")
    except Exception as e:
        print(f"{RD}  Error guardando JSON: {e}{R}")

    return png_out


# ═══════════════════════════════════════════════════════
#  BUCLE PRINCIPAL
# ═══════════════════════════════════════════════════════
def run_sesion(modelo_ollama):
    print("\n" + "="*58)
    print(f"{CY}  MIDIENDO: {GR}{modelo_ollama}{R}")
    print(f"  {DM}Las metricas analizan las RESPUESTAS del modelo.{R}")
    print(f"  {DM}Escribe 'salir' o '/chao' para terminar la sesion.{R}")
    print("="*58)

    historial_completo = []   # [user, ia, user, ia, ...]
    respuestas_ia      = []   # solo respuestas del modelo
    C_n       = 0.0
    integral  = 0.0
    lambda_d  = 0.15
    R_0       = 1.0
    metricas  = []
    tiempos   = []

    while True:
        try:
            user_input = input(f"\n{MG}Tu:{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if user_input.lower() in ["salir", "exit", "/chao", "/menu"]:
            break
        if not user_input:
            continue

        t_actual = time.time()
        tiempos.append(t_actual)
        delta_t = 0 if len(tiempos) < 2 else tiempos[-1] - tiempos[-2]

        # ── 1. Primero generar la respuesta del modelo ────────
        historial_completo.append(user_input)
        print(f"{DM}  [generando respuesta con {modelo_ollama}...]{R}", end="\r")
        respuesta, error = generar_respuesta_ollama(modelo_ollama, historial_completo)

        if error:
            print(f"{RD}  Error Ollama: {error}{R}")
            respuesta = "Interesante. Continua con tu idea."

        historial_completo.append(respuesta)
        print(f"\n{GR}{modelo_ollama}:{R} {respuesta}\n")

        # ── 2. Calcular metricas sobre la RESPUESTA del modelo ─
        print(f"{DM}  [analizando respuesta del modelo...]{R}", end="\r")

        contexto = " ".join(historial_completo[:-1])  # todo excepto la ultima respuesta

        A           = calcular_A(respuesta, respuestas_ia)
        E_n         = calcular_perplejidad(contexto, respuesta)
        D_KL        = calcular_D_KL(contexto, respuesta)
        novedad_sem = calcular_novedad_semantica(contexto, respuesta)
        long_norm   = calcular_longitud_norm(respuesta)

        R_t       = R_0 * np.exp(-lambda_d * delta_t)
        integral  = integral * np.exp(-lambda_d * delta_t) + R_t * D_KL
        modulador = 1.0 / (1.0 + np.exp(-np.clip(A * E_n, -50, 50)))
        C_n       = modulador * integral

        emo_score = 0.5
        if sentiment_analyzer:
            try:
                res = sentiment_analyzer(respuesta[:512])[0]
                emo_score = res["score"] if res["label"] == "POSITIVE" else 1.0 - res["score"]
            except Exception:
                pass

        respuestas_ia.append(respuesta)
        metricas.append({
            "interaccion":       len(metricas) + 1,
            "user_input":        user_input,
            "respuesta_modelo":  respuesta,
            "A":                 A,
            "E_n":               E_n,
            "D_KL":              D_KL,
            "novedad_semantica": novedad_sem,
            "longitud_norm":     long_norm,
            "R_t":               R_t,
            "modulador":         modulador,
            "C_n+1":             C_n,
            "emo_score":         emo_score,
            "t":                 t_actual,
        })

        # ── 3. Mostrar metricas en consola ─────────────────────
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
            print(f"{DM}  >> Respuestas predecibles. Intenta un estimulo mas disruptivo.{R}")

    # ── Resumen en consola ────────────────────────────────
    print("\n" + "="*58)
    print(f"{CY}  SESION FINALIZADA -- {modelo_ollama}{R}")
    print(f"  Plasticidad final C_n:  {BL}{C_n:.4f}{R}")
    print(f"  Total interacciones:    {WH}{len(metricas)}{R}")
    if metricas:
        vals = [m["C_n+1"] for m in metricas]
        print(f"  Plasticidad media:      {WH}{np.mean(vals):.4f}{R}")
        print(f"  Plasticidad maxima:     {GR}{max(vals):.4f}{R} (#{np.argmax(vals)+1})")
        print(f"  Plasticidad minima:     {RD}{min(vals):.4f}{R} (#{np.argmin(vals)+1})")
    print("="*58)

    # ── Generar informe ───────────────────────────────────
    if metricas:
        print(f"\n{CY}  Generando informe con graficos...{R}")
        ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        ruta = generar_informe(modelo_ollama, metricas, ts)
        if ruta:
            print(f"{GR}  Informe PNG listo: {ruta}{R}")
            print(f"{DM}  Abrelo con cualquier visor de imagenes.{R}")
    else:
        print(f"{DM}  Sin datos para generar informe.{R}")


# ═══════════════════════════════════════════════════════
#  ENTRADA PRINCIPAL
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{CY}  PLASTICIDAD ADAPTATIVA v3.0{R}")
    print(f"  {DM}Mide la adaptabilidad de los modelos Ollama.{R}")
    print(f"  {DM}Las metricas se calculan sobre las RESPUESTAS del modelo.{R}")

    while True:
        modelo = menu_seleccion_modelo()

        if modelo is None:
            print(f"\n{GR}  Hasta pronto, German el Gris!{R}\n")
            break

        run_sesion(modelo)
        print(f"\n{DM}  [Volviendo al menu...]{R}")
