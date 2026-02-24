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
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════
#  CONFIGURACIÓN Y MODELOS
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
#  FUNCIONES DE APOYO (LÓGICA ORIGINAL)
# ═══════════════════════════════════════════════════════
def obtener_modelos_ollama():
    try:
        r = requests.get(OLLAMA_TAGS, timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except:
        return []

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
    except Exception as e:
        return None, str(e)

# ═══════════════════════════════════════════════════════
#  CARGAR MODELOS DE ANÁLISIS
# ═══════════════════════════════════════════════════════
print("Cargando modelos de análisis (GPT2 + Embeddings)...")
tokenizer  = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except:
    sentiment_analyzer = None

# ═══════════════════════════════════════════════════════
#  MÉTRICAS
# ═══════════════════════════════════════════════════════
def calcular_novedad_semantica(contexto, respuesta):
    if not contexto.strip() or not respuesta.strip(): return 0.0
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
    if len(new_ids) == 0: return 1.0
    with torch.no_grad():
        out = gpt2_model(**inputs)
        logits = out.logits[:, ctx_len-1:full_len-1, :] if contexto.strip() else out.logits[:, 0:full_len-1, :]
    log_p = torch.log_softmax(logits, dim=-1)
    if logits.shape[1] != len(new_ids):
        logits = out.logits[:, -len(new_ids):, :]
        log_p  = torch.log_softmax(logits, dim=-1)
    idx = new_ids.unsqueeze(0).unsqueeze(-1)
    tlp = log_p.gather(2, idx).squeeze(-1)
    return float(np.exp(-tlp.mean().item()))

def calcular_A(respuesta, respuestas_previas):
    tokens_n = set(tokenizer.encode(respuesta))
    tokens_p = set()
    for r in respuestas_previas: tokens_p.update(tokenizer.encode(r))
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

# ═══════════════════════════════════════════════════════
#  INTERPRETACIONES DE MÉTRICAS
# ═══════════════════════════════════════════════════════
METRIC_INTERPRETATIONS = {
    "C_n": {
        "name": "Plasticidad Adaptativa",
        "desc": "Capacidad del modelo para adaptarse y aprender de la conversación.",
        "high": "El modelo está aprendiendo y adaptándose bien, mostrando respuestas novedosas y relevantes.",
        "low": "El modelo está estancado, repitiendo ideas o siendo predecible."
    },
    "A": {
        "name": "Factor A (Novedad de Vocabulario)",
        "desc": "Mide cuántas palabras nuevas introduce el modelo en su respuesta en comparación con el historial.",
        "high": "El modelo está usando vocabulario fresco, lo que indica creatividad.",
        "low": "El modelo está reciclando palabras, lo que sugiere falta de originalidad."
    },
    "E_n": {
        "name": "Creatividad (Perplejidad)",
        "desc": "Indica cuán 'sorprendente' o inesperada es la respuesta del modelo, pero aún coherente.",
        "high": "La respuesta es muy creativa y poco predecible.",
        "low": "La respuesta es muy predecible y poco original."
    },
    "D_KL": {
        "name": "Impacto (Divergencia KL)",
        "desc": "Mide cuánto cambia la 'visión del mundo' del modelo después de su respuesta. Un valor alto indica un cambio significativo.",
        "high": "La respuesta del modelo ha introducido una nueva perspectiva o ha cambiado drásticamente el flujo de la conversación.",
        "low": "La respuesta es continuista y no aporta grandes cambios."
    },
    "novedad": {
        "name": "Novedad Semántica",
        "desc": "Cuán diferente es el significado de la respuesta del modelo respecto al contexto previo.",
        "high": "La respuesta explora un tema nuevo o un ángulo diferente.",
        "low": "La respuesta se mantiene muy cerca del tema anterior."
    }
}

# ═══════════════════════════════════════════════════════
#  INTERFAZ GRÁFICA (GUI)
# ═══════════════════════════════════════════════════════
class PlasticityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plasticidad Adaptativa v3.0 - GUI")
        self.root.geometry("1000x800")
        self.root.configure(bg="#1e1e1e")

        # Variables de estado
        self.modelo_seleccionado = tk.StringVar()
        self.historial_completo = []
        self.respuestas_ia = []
        self.metricas = []
        self.tiempos = []
        self.C_n = 0.0
        self.integral = 0.0
        self.lambda_d = 0.15
        self.R_0 = 1.0

        self.setup_ui()

    def setup_ui(self):
        # Estilo
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#3a3a3a"), ("!disabled", "#007acc")])
        style.configure("TCombobox", fieldbackground="#3c3c3c", background="#3c3c3c", foreground="#ffffff")
        style.map("TCombobox", fieldbackground=[("readonly", "#3c3c3c")], selectbackground=[("readonly", "#007acc")])

        # Panel de Selección de Modelo
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Seleccionar Modelo:").pack(side=tk.LEFT, padx=5)
        modelos = obtener_modelos_ollama()
        if not modelos: modelos = ["mistral", "llama3.1:8b", "gemma3:4b"]
        self.combo_modelo = ttk.Combobox(top_frame, textvariable=self.modelo_seleccionado, values=modelos, width=30, state="readonly")
        self.combo_modelo.pack(side=tk.LEFT, padx=5)
        self.combo_modelo.set(modelos[0])

        self.btn_iniciar = ttk.Button(top_frame, text="Iniciar Sesión", command=self.iniciar_sesion)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)

        self.btn_stats = ttk.Button(top_frame, text="Ver Estadísticas", command=self.mostrar_estadisticas, state=tk.DISABLED)
        self.btn_stats.pack(side=tk.RIGHT, padx=5)

        # Área de Chat
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, bg="#252526", fg="#d4d4d4", font=("Consolas", 11), state=tk.DISABLED)
        self.chat_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # Panel de Entrada
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.pack(fill=tk.X)

        self.user_input = ttk.Entry(input_frame, font=("Segoe UI", 11), background="#3c3c3c", foreground="#ffffff", insertbackground="#ffffff")
        self.user_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.user_input.bind("<Return>", lambda e: self.enviar_mensaje())

        self.btn_enviar = ttk.Button(input_frame, text="Enviar", command=self.enviar_mensaje, state=tk.DISABLED)
        self.btn_enviar.pack(side=tk.LEFT, padx=5)

        # Barra de Estado (Plasticidad)
        self.status_frame = ttk.Frame(self.root, padding="5")
        self.status_frame.pack(fill=tk.X)
        self.plasticity_label = ttk.Label(self.status_frame, text="Plasticidad C_n: 0.0000 | Estado: -")
        self.plasticity_label.pack(side=tk.LEFT)

    def log_chat(self, sender, message, color="#ffffff"):
        self.chat_area.config(state=tk.NORMAL)
        tag = sender.lower().replace(" ", "_")
        self.chat_area.tag_configure(tag, foreground=color, font=("Consolas", 11, "bold"))
        self.chat_area.insert(tk.END, f"{sender}: ", tag)
        self.chat_area.insert(tk.END, f"{message}\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def iniciar_sesion(self):
        modelo = self.modelo_seleccionado.get()
        if not modelo:
            messagebox.showwarning("Error", "Selecciona un modelo primero.")
            return
        
        self.historial_completo = []
        self.respuestas_ia = []
        self.metricas = []
        self.tiempos = []
        self.C_n = 0.0
        self.integral = 0.0
        
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state=tk.DISABLED)
        
        self.log_chat("SISTEMA", f"Sesión iniciada con {modelo}", "#00ff00")
        self.btn_enviar.config(state=tk.NORMAL)
        self.btn_stats.config(state=tk.NORMAL)
        self.user_input.focus()

    def enviar_mensaje(self):
        msg = self.user_input.get().strip()
        if not msg: return
        
        self.user_input.delete(0, tk.END)
        self.log_chat("Tú", msg, "#569cd6")
        
        modelo = self.modelo_seleccionado.get()
        self.historial_completo.append(msg)
        
        t_actual = time.time()
        self.tiempos.append(t_actual)
        delta_t = 0 if len(self.tiempos) < 2 else self.tiempos[-1] - self.tiempos[-2]

        # Simular procesamiento
        self.root.config(cursor="watch")
        self.root.update()
        
        respuesta, error = generar_respuesta_ollama(modelo, self.historial_completo)
        
        if error:
            respuesta = f"[Error Ollama: {error}] Interesante. Continua con tu idea."
        
        self.log_chat(modelo, respuesta, "#4ec9b0")
        self.historial_completo.append(respuesta)
        
        # Análisis de Métricas
        contexto = " ".join(self.historial_completo[:-1])
        A = calcular_A(respuesta, self.respuestas_ia)
        E_n = calcular_perplejidad(contexto, respuesta)
        D_KL = calcular_D_KL(contexto, respuesta)
        novedad_sem = calcular_novedad_semantica(contexto, respuesta)
        
        R_t = self.R_0 * np.exp(-self.lambda_d * delta_t)
        self.integral = self.integral * np.exp(-self.lambda_d * delta_t) + R_t * D_KL
        modulador = 1.0 / (1.0 + np.exp(-np.clip(A * E_n, -50, 50)))
        self.C_n = modulador * self.integral
        
        self.respuestas_ia.append(respuesta)
        self.metricas.append({
            "interaccion": len(self.metricas) + 1,
            "A": A, "E_n": E_n, "D_KL": D_KL, "C_n": self.C_n, "novedad": novedad_sem
        })
        
        status_text = ""
        status_color = "#ffffff"
        if self.C_n >= 0.7: # Umbral alto para plasticidad
            status_text = "ADAPTACIÓN ACTIVA: El modelo está aprendiendo y adaptándose muy bien."
            status_color = "#00ff00"
        elif self.C_n >= 0.5:
            status_text = "ADAPTACIÓN MODERADA: El modelo muestra buena capacidad de adaptación."
            status_color = "#90ee90"
        else:
            status_text = "ESTANCAMIENTO: El modelo es predecible o repite ideas."
            status_color = "#ff0000"

        self.plasticity_label.config(text=f"Plasticidad C_n: {self.C_n:.4f} | Estado: {status_text}", foreground=status_color)
        
        self.root.config(cursor="")

    def mostrar_estadisticas(self):
        if not self.metricas:
            messagebox.showinfo("Estadísticas", "No hay datos suficientes para mostrar estadísticas.")
            return

        stats_win = tk.Toplevel(self.root)
        stats_win.title(f"Estadísticas de Sesión - {self.modelo_seleccionado.get()}")
        stats_win.geometry("1200x800") # Aumentar tamaño para las descripciones
        stats_win.configure(bg="#1e1e1e")

        fig = plt.figure(figsize=(12, 7), dpi=100, facecolor=\'#1e1e1e\')
        gs = GridSpec(2, 2, figure=fig)
        
        plt.rcParams[\'text.color\'] = \'white\'
        plt.rcParams[\'axes.labelcolor\'] = \'white\'
        plt.rcParams[\'xtick.color\'] = \'white\'
        plt.rcParams[\'ytick.color\'] = \'white\'
        plt.rcParams[\'axes.edgecolor\'] = \'#444444\'
        plt.rcParams[\'grid.color\'] = \'#444444\'

        x = [m["interaccion"] for m in self.metricas]
        
        # Función auxiliar para añadir descripciones
        def add_description(ax, metric_key):
            interpretation = METRIC_INTERPRETATIONS.get(metric_key, {})
            name = interpretation.get("name", "")
            desc = interpretation.get("desc", "")
            high_desc = interpretation.get("high", "")
            low_desc = interpretation.get("low", "")
            
            full_desc = f"**{name}:** {desc}\n\n"
            if high_desc: full_desc += f"  - **Valores Altos:** {high_desc}\n"
            if low_desc: full_desc += f"  - **Valores Bajos:** {low_desc}\n"
            
            ax.text(0.02, 0.98, full_desc, transform=ax.transAxes, fontsize=9, verticalalignment=\'top\', 
                    bbox=dict(boxstyle=\'round,pad=0.5\', fc=\'#2d2d2d\', ec=\'#444444\', lw=1, alpha=0.8), 
                    color=\'white\', wrap=True)

        # 1. Evolución de Plasticidad (C_n)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(\'#252526\')
        ax1.plot(x, [m["C_n"] for m in self.metricas], marker=\'o\', color=\'#4ec9b0\', linewidth=2)
        ax1.set_title("Evolución de Plasticidad (C_n)")
        ax1.set_xlabel("Interacción")
        ax1.set_ylabel("Valor C_n")
        ax1.grid(True, alpha=0.2)
        add_description(ax1, "C_n")

        # 2. Creatividad vs Impacto
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(\'#252526\')
        ax2.plot(x, [m["E_n"] for m in self.metricas], marker=\'s\', label="Creatividad (E_n)", color=\'#569cd6\')
        ax2.plot(x, [m["D_KL"] for m in self.metricas], marker=\'^\', label="Impacto (D_KL)", color=\'#ce9178\')
        ax2.set_title("Creatividad vs Impacto")
        ax2.set_xlabel("Interacción")
        ax2.set_ylabel("Valor")
        ax2.legend(facecolor=\'#252526\', edgecolor=\'#444444\', labelcolor=\'white\')
        ax2.grid(True, alpha=0.2)
        add_description(ax2, "E_n")
        add_description(ax2, "D_KL")

        # 3. Factor A (Vocabulario Nuevo)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(\'#252526\')
        ax3.bar(x, [m["A"] for m in self.metricas], color=\'#dcdcaa\')
        ax3.set_title("Factor A (Novedad de Vocabulario)")
        ax3.set_xlabel("Interacción")
        ax3.set_ylabel("Valor A")
        ax3.grid(True, alpha=0.2)
        add_description(ax3, "A")

        # 4. Novedad Semántica
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(\'#252526\')
        ax4.fill_between(x, [m["novedad"] for m in self.metricas], color=\'#c586c0\', alpha=0.3)
        ax4.plot(x, [m["novedad"] for m in self.metricas], color=\'#c586c0\')
        ax4.set_title("Novedad Semántica")
        ax4.set_xlabel("Interacción")
        ax4.set_ylabel("Valor Novedad")
        ax4.grid(True, alpha=0.2)
        add_description(ax4, "novedad")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=stats_win)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlasticityApp(root)
    root.mainloop()
