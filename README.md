# 🧠 Vision-Language Model para Generación de Informes Clínicos Radiológicos

Proyecto de Fin de Grado en Ingeniería Informática (ETSIIT - UGR)

Autor: Rafael Carrillo Arroyo

---

##  Descripción del Proyecto

Este proyecto implementa un **modelo multimodal Vision-Language (VLM)** capaz de generar informes clínicos radiológicos a partir de imágenes de rayos X de tórax.

El sistema combina:

-  **Codificador visual (DenseNet121)** para extracción de características radiológicas
-  **Decodificador lingüístico (GPT-2)** para generación de texto clínico
-  **Adaptación eficiente mediante LoRA (Low-Rank Adaptation)**

El objetivo es estudiar la capacidad de modelos multimodales para asistir en la redacción automática de informes médicos manteniendo coherencia clínica y estabilidad lingüística.

---

##  Arquitectura del Sistema

El modelo se compone de tres bloques principales:

### 1. Encoder Visual
- Backbone: `DenseNet121 (PyTorch)`
- Entrada: imágenes de rayos X (escala de grises)
- Salida: embedding clínico latente
- Fine-tuning: congelado tras entrenamiento supervisado

### 2. Fusión Multimodal (implícita)
- Proyección de features visuales hacia espacio compatible con el decoder

### 3. Decoder Lingüístico
- Modelo base: `GPT-2 Small`
- Adaptación: `LoRA (PEFT)`
- Generación: texto clínico estructurado (Findings / Impression)
---

## 🧬 Técnica Principal: LoRA

Se aplica **Low-Rank Adaptation (LoRA)** para:

- Reducir el número de parámetros entrenables
- Evitar sobreajuste del modelo lingüístico
- Mantener la capacidad generativa original de GPT-2
- Permitir adaptación eficiente del decoder sin modificar los pesos base del modelo

---