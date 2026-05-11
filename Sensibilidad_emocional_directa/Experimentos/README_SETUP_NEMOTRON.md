# README — Configuración para correr Nemotron 70B local con CUDA

## Requisitos de hardware

| Componente | Mínimo recomendado |
|---|---|
| GPU NVIDIA | RTX 3090 / A100 (≥ 24 GB VRAM) |
| RAM del sistema | 64 GB |
| Almacenamiento | ~45 GB libres para el modelo |
| Driver NVIDIA | ≥ 525.xx |
| CUDA Toolkit | ≥ 12.1 |

> **Nota:** Nemotron 70B en formato Q4 necesita aproximadamente 40-42 GB de VRAM.  
> Para GPUs de < 24 GB puedes usar quantización más agresiva (Q2/Q3) o repartir  
> la carga entre varias GPUs con `OLLAMA_NUM_GPU=2`.

---

## 1. Instalar Ollama (con soporte CUDA automático)

Ollama detecta la GPU NVIDIA automáticamente si los drivers están instalados.

### Linux (recomendado para servidores con GPU)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Descarga el instalador desde: https://ollama.com/download/windows  
El instalador configura CUDA automáticamente.

---

## 2. Verificar que Ollama usa la GPU

```bash
# Iniciar el servidor Ollama (en segundo plano)
ollama serve &

# Verificar que detecta la GPU
ollama run nemotron:70b "Hola"
# En los logs debe aparecer: "using CUDA device: ..."
```

También puedes verificar el uso de VRAM con:
```bash
nvidia-smi
```

---

## 3. Descargar el modelo Nemotron 70B

```bash
ollama pull nemotron:70b
```

> La descarga pesa aproximadamente 40-45 GB. Asegúrate de tener buena conexión  
> o transfiere el modelo manualmente (ver sección 5).

---

## 4. Instalar dependencias de Python

```bash
pip install ollama==0.6.1 pandas==2.2.3
```

O usando el requirements.txt del proyecto:
```bash
pip install -r requirements.txt
```

---

## 5. Transferir el modelo sin re-descargar (opcional)

Si ya tienes el modelo descargado en otro ordenador, puedes copiarlo  
directamente para evitar volver a descargarlo.

**Ubicación de los modelos en Ollama:**

| OS | Ruta |
|---|---|
| Linux | `~/.ollama/models/` |
| Windows | `C:\Users\<usuario>\.ollama\models\` |
| macOS | `~/.ollama/models/` |

Copia la carpeta `models/` completa al mismo directorio en el ordenador destino.

---

## 6. Correr el experimento

```bash
# Asegúrate de que Ollama está corriendo
ollama serve

# En otra terminal, ejecutar el experimento
python experimento_sensibilidad_nemotron70b_50.py
```

El CSV de resultados se guardará en:
```
Experimentos/dataset_experimento/sensibilidad_nemotron_70b_50iter.csv
```

---

## 7. Variables de entorno útiles

```bash
# Usar múltiples GPUs (si tienes más de una)
export OLLAMA_NUM_GPU=2

# Forzar uso de GPU específica
export CUDA_VISIBLE_DEVICES=0

# Aumentar contexto si es necesario
export OLLAMA_NUM_CTX=4096
```

En **Windows PowerShell**:
```powershell
$env:OLLAMA_NUM_GPU = "2"
$env:CUDA_VISIBLE_DEVICES = "0"
```

---

## 8. Solución de problemas

| Problema | Solución |
|---|---|
| `model not found` | Ejecuta `ollama pull nemotron:70b` |
| `CUDA out of memory` | Usa quantización menor: `ollama pull nemotron:70b-q2_K` |
| Ollama no detecta GPU | Verifica drivers: `nvidia-smi` debe mostrar la GPU |
| Respuestas lentas | Reduce `num_predict` en `PARAMS` del script |
| `connection refused` | Inicia el servidor: `ollama serve` |
