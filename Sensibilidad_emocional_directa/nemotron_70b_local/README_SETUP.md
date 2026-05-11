# Nemotron 70B — Configuración local con CUDA

## Contenido de esta carpeta

```
nemotron_70b_local/
├── experimento_sensibilidad_nemotron70b_50.py   ← script principal
├── requirements.txt                              ← dependencias Python
├── README_SETUP.md                               ← este archivo
└── dataset_experimento/                          ← se crea al correr el script
    └── sensibilidad_nemotron_70b_50iter.csv
```

---

## Requisitos de hardware

| Componente | Mínimo recomendado |
|---|---|
| GPU NVIDIA | RTX 3090 / A100 (≥ 24 GB VRAM) |
| RAM del sistema | 64 GB |
| Almacenamiento libre | ~45 GB para el modelo |
| Driver NVIDIA | ≥ 525.xx |
| CUDA Toolkit | ≥ 12.1 |

> Nemotron 70B en Q4 necesita ~40-42 GB de VRAM.  
> Con varias GPUs usa `OLLAMA_NUM_GPU=2` (ver sección 5).

---

## Paso 1 — Instalar Ollama (detecta CUDA automáticamente)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**  
Descarga desde https://ollama.com/download/windows  
El instalador configura CUDA solo.

---

## Paso 2 — Descargar el modelo

```bash
ollama pull nemotron:70b
```

> ~45 GB de descarga. Alternativamente, copia la carpeta de modelos  
> desde otro ordenador (ver sección 6).

---

## Paso 3 — Instalar dependencias Python

```bash
pip install -r requirements.txt
```

---

## Paso 4 — Correr el experimento

```bash
# Terminal 1: iniciar el servidor Ollama
ollama serve

# Terminal 2: ejecutar el experimento
python experimento_sensibilidad_nemotron70b_50.py
```

El CSV se guardará automáticamente en `dataset_experimento/`.

---

## Paso 5 — Variables de entorno útiles

```bash
# Linux / macOS
export OLLAMA_NUM_GPU=2        # usar 2 GPUs
export CUDA_VISIBLE_DEVICES=0  # elegir GPU específica

# Windows PowerShell
$env:OLLAMA_NUM_GPU = "2"
$env:CUDA_VISIBLE_DEVICES = "0"
```

---

## Paso 6 — Transferir el modelo sin re-descargar

Copia la carpeta de modelos de Ollama al ordenador destino:

| OS | Ruta |
|---|---|
| Linux / macOS | `~/.ollama/models/` |
| Windows | `C:\Users\<usuario>\.ollama\models\` |

---

## Solución de problemas

| Problema | Solución |
|---|---|
| `model not found` | `ollama pull nemotron:70b` |
| `CUDA out of memory` | Prueba `ollama pull nemotron:70b-q2_K` |
| Ollama no detecta GPU | Verifica con `nvidia-smi` |
| `connection refused` | Ejecuta `ollama serve` primero |
