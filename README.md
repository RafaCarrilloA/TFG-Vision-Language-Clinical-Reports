# 🧠 Vision-Language Model para Generación de Informes Clínicos Radiológicos

Proyecto de Fin de Grado en Ingeniería Informática (ETSIIT - UGR)

**Autor:** Rafael Carrillo Arroyo

---

## Descripción del Proyecto

Este proyecto implementa un **modelo multimodal Vision-Language (VLM)** capaz de generar informes clínicos radiológicos a partir de imágenes de rayos X de tórax.

El sistema combina:

* **Codificador visual (DenseNet121)** para extracción de características radiológicas.
* **Decodificador lingüístico (GPT-2)** para generación de texto clínico.
* **Adaptación eficiente mediante LoRA (Low-Rank Adaptation)**.

El objetivo es estudiar la capacidad de modelos multimodales para asistir en la redacción automática de informes médicos manteniendo coherencia clínica y estabilidad lingüística.

---

## Recursos del Proyecto

Debido al tamaño de los datasets y de los modelos entrenados, los recursos completos del proyecto se encuentran disponibles en Google Drive:

**Google Drive del proyecto:**

https://drive.google.com/drive/folders/190Xspevq_DuxQ3TelS3kAR5PC_xw6rMz?usp=sharing

La carpeta incluye:

* `data/` → Datasets utilizados durante el desarrollo.
* `checkpoints/` → Modelos entrenados y pesos generados.
* `assets/` → Recursos auxiliares del proyecto.
* `notebooks/` → Versiones completas de los cuadernos de trabajo.

---

## Arquitectura del Sistema

El modelo se compone de tres bloques principales:

### 1. Encoder Visual

* Backbone: `DenseNet121 (PyTorch)`
* Entrada: imágenes de rayos X (escala de grises)
* Salida: embedding clínico latente
* Fine-tuning: congelado tras entrenamiento supervisado

### 2. Fusión Multimodal (implícita)

* Proyección de características visuales hacia un espacio compatible con el decoder lingüístico.

### 3. Decoder Lingüístico

* Modelo base: `GPT-2 Small`
* Adaptación: `LoRA (PEFT)`
* Generación: texto clínico estructurado (*Findings* / *Impression*)

---

## 🧬 Técnica Principal: LoRA

Se aplica **Low-Rank Adaptation (LoRA)** para:

* Reducir el número de parámetros entrenables.
* Evitar sobreajuste del modelo lingüístico.
* Mantener la capacidad generativa original de GPT-2.
* Permitir adaptación eficiente del decoder sin modificar los pesos base del modelo.

---

## Estructura del Repositorio

```text
TFG-Vision-Language-Clinical-Reports/
│
├── notebooks/
├── assets/
├── checkpoints/
├── data/
├── outputs/
├── README.md
└── .gitignore
```

---

## Tecnologías Utilizadas

* Python
* PyTorch
* MONAI
* Transformers (Hugging Face)
* GPT-2
* PEFT / LoRA
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

## Autor

**Rafael Carrillo Arroyo**

Grado en Ingeniería Informática
Escuela Técnica Superior de Ingenierías Informática y de Telecomunicación (ETSIIT)
Universidad de Granada (UGR)
