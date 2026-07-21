# 🧠 Vision-Language Model para Generación de Informes Clínicos Radiológicos

Proyecto de Fin de Grado en Ingeniería Informática (ETSIIT - UGR)  
**Autor:** Rafael Carrillo Arroyo

---

## 📋 Resumen Ejecutivo
Este proyecto implementa un pipeline multimodal avanzado que fusiona visión por computador y procesamiento de lenguaje natural (NLP) para la redacción autorregresiva de informes radiológicos. El sistema aborda los desafíos del **sesgo de normalidad** y la **desalineación de dominio** mediante una arquitectura dividida en tres fases de ingeniería estricta.

---

## 🏗️ Arquitectura del Sistema: Pipeline de 3 Fases
### FASE 1: Codificador Visual Clínico (Extractor Espacial)
**Objetivo:** Construcción de unos cimientos visuales universales y robustos que codifiquen la anatomía radiológica preservando estrictamente la topología espacial, generando el vocabulario base para el modelo generativo.

**Estrategia Arquitectónica y de Optimización:**
* *Backbone Médico:* Adaptación de una red **DenseNet121** mediante *Weight Surgery* (trasplante de pesos preentrenados y adaptación a entrada monocanal).
* *Preservación Topológica:* Se extirpa deliberadamente la capa de *Global Average Pooling* extrayendo una **cuadrícula espacial de 10x10** (100 tokens visuales continuos de 1024 canales) capaz de aislar regiones anatómicas concretas.
* *Blindaje Estadístico:* Aprendizaje multietiqueta en paralelo (14 patologías) combatiendo el desbalanceo extremo (*Long-Tail*) mediante pérdida asimétrica ponderada (BCE + Pos-Weight) y mapeo de incertidumbre clínico (*Soft Targets*).

**Rendimiento Global y Robustez (Cohorte de Test):**
El modelo demuestra una alta capacidad de separación estadística global (ROC), manteniendo la robustez estructural frente al desbalanceo masivo de clases sanas, como certifica la curva PR. 

<div align="center">
  <img src="assets/Resultados_Evaluacion/Modulo_1/test/curvas_roc_test_finales.png" width="48%" alt="Curva ROC Conjunta Multietiqueta">
  <img src="assets/Resultados_Evaluacion/Modulo_1/test/curvas_pr_test_finales.png" width="48%" alt="Curva PR Conjunta Multietiqueta">
</div>

**Rendimiento Neto Clínico (F1-Score):**
El análisis del F1-Score sobre el Test Set confirma un rendimiento de grado clínico en patologías sólidas y derrames, donde el modelo exhibe su mayor fiabilidad diagnóstica.

<div align="center">
  <img src="assets/Resultados_Evaluacion/Modulo_1/test/f1_scores_test.png" width="75%" alt="Gráfico de barras F1-Score">
</div>

📄 **[Ver tabla completa de métricas clínicas y umbrales (CSV)](assets/Resultados_Evaluacion/Modulo_1/test/metricas_clinicas_test.csv)**

> **Explicabilidad y Auditoría Visual (Grad-CAM):**
> ![Showcase Grad-CAM](assets/Resultados_Evaluacion/Modulo_1/test/GradCAM/gradcam_showcase_readme.png)
> *El mapeo térmico de activación certifica que la DenseNet121 localiza la patología real (ej. Derrame Pleural, Edema, Atelectasia) de forma matemática, evadiendo el "Shortcut Learning" o sesgo de instrumental.*
---
### FASE 2: Puente Multimodal y Alineamiento Geométrico (Alineamiento Visual-Lingüístico Pure)
**Objetivo:** Establecer el nexo de unión entre el extractor visual congelado y el decodificador de lenguaje (BioGPT). Esta fase opera bajo el protocolo de "Alineamiento Puro": el LLM se mantiene estrictamente congelado, y toda la responsabilidad de proyectar la semántica visual hacia el espacio sintáctico del LLM recae en un proyector multimodal (MLP) asistido por inyección de Embeddings Posicionales 2D aprendidos.

**Éxito de Ingeniería y Auditoría Matemática (Cohorte Test $N=380$):**
El diseño del puente multimodal (MLP + LayerNorm + Centrado Dinámico) logró estabilizar perfectamente la matemática de la inyección visual sin saturar la red receptora.
* **Inmutabilidad del LLM:** $\text{Max-Diff} = 0.00$ (Certificación total de BioGPT congelado).
* **Estabilidad Energética:** Norma L2 estabilizada en **32.22** (Previene la saturación matemática de la capa Softmax).
* **Salud del Espacio Latente (Mitigación del "Efecto Cono"):** La similitud coseno centrada se desplomó a un **0.0005**, logrando una dispersión isótropa óptima y destruyendo la anisotropía nativa de los LLMs.

> **Demostración Geométrica de la Coherencia Espacial:**
> <div align="center">
>   <img src="assets/Resultados_Evaluacion/Modulo_2/post_auditoria_topologica_2d.png" width="60%" alt="Matriz de Afinidad Topológica pos_embeddings">
> </div>
> 
> *Esta matriz de afinidad demuestra empíricamente que el modelo ha aprendido una topología bidimensional sobre una secuencia plana. Las bandas paralelas a la diagonal principal confirman correlaciones geométricas (ej. correlación entre el token físico 5 y el 15, o el 25), lo que demuestra que el proyector entiende la anatomía de una radiografía 10x10 sin supervisión explícita.*

**La Paradoja del "Proyector Puro": Confirmación de la Inercia Lingüística**
A pesar de la impecable ejecución geométrica y de ingeniería, la evaluación clínica revela una **Tasa Macro de Captura Patológica crítica del 6.69%**. Al estar BioGPT congelado, su fuerte *prior* lingüístico de preentrenamiento domina por completo sobre la señal visual, forzando al modelo a redactar reportes de normalidad genéricos incluso ante lesiones severas.

> **⚠️ Ejemplo de Alucinación por "Sesgo de Normalidad" (Paciente con fracturas y derrame):**
> * **[REAL]:** *"Findings: Cardiomediastinal contours are unchanged. There are stable fractures... Pleural effusion..."*
> * **[GENERADO]:** *"Normal, the heart is in normal site. No cardiomegaly or pleural effusion. Impression: The chest is clear."*

**Veredicto Metodológico:** El alineamiento puro es incapaz de romper la inercia lingüística del LLM. Este descubrimiento empírico valida la hipótesis de la memoria y hace metodológicamente imperativa la transición a la **FASE 3 (Inyección de LoRA)** para intervenir la corteza de atención de BioGPT y recuperar la sensibilidad diagnóstica médica.

---

### FASE 3: Sincronización Clínica Autorregresiva (LoRA y Focal Loss)
**Objetivo:** Superar el Sesgo de Normalidad de la Fase 2 dotando a **BioGPT** de adaptabilidad diagnóstica fina, aprovechando su mente médica preentrenada (PubMed) sin destruir la geometría del espacio latente estabilizado.

**Estrategia Quirúrgica:**
* **Intercepción PEFT/LoRA:** Rango ampliado ($r=32, \alpha=64$) dirigido a la corteza atencional (`q_proj`, `k_proj`, `v_proj`, `out_proj`), ajustando solo un $\approx 1.5\% - 2\%$ de los parámetros para prevenir el olvido catastrófico.
* **Estabilización del Gradiente:** Uso de **Focal Loss Autorregresiva** ($\gamma=2.0, \alpha=1.0$) para destruir la inercia estadística de los tokens sanos.

**Resultados Consolidados (Cohorte Ciega $N=380$):**
* **Soberanía Visual:** Índice de Dependencia Visual (**Macro-VDI**) de **0.8333**. El 83.33% de la variabilidad lexicográfica está gobernada por los píxeles.
* **Disociación Metrológica NLG:** Convergencia semántica profunda alcanzando un **BERTScore F1 de 0.8777**.
* **Relevancia Clínica (BART Zero-Shot NLI):** Rendimiento SOTA en patologías sólidas: Opacidad Pulmonar (**78.94%**), Consolidación (**73.03%**) y Alteraciones Pleurales (**71.08%**). 

> **Convergencia y Reportes Clínicos (NLG & NLI):**
> ![Convergencia Fase 3](assets/Resultados_Evaluacion/Modulo_3/convergencia_fase3_lora.png)
> *Tablas detalladas de la evaluación final:*
> - 📄 [Ver Métricas de Generación NLG (BLEU, ROUGE, BERTScore)](assets/Resultados_Evaluacion/Modulo_3/metricas_generacion_nlg.csv)
> - 🏥 [Ver Resultados de la Auditoría Clínica BART NLI](assets/Resultados_Evaluacion/Modulo_3/resultados_auditoria_F1_BART_fase3.csv)
> - 🔬 [Ver Métricas Clínicas Finales](assets/Resultados_Evaluacion/Modulo_3/metricas_clinicas_bart.csv)

---

## 📂 Recursos y Estructura del Repositorio

🔗 **[Acceso al Google Drive del Proyecto (Datasets y Pesos)](https://drive.google.com/drive/folders/190Xspevq_DuxQ3TelS3kAR5PC_xw6rMz?usp=sharing)**

```text
TFG-Vision-Language-Clinical-Reports/
├── notebooks/       # Cuadernos modulares ejecutables (Fase 1, 2 y 3)
├── assets/          #
│   ├── Resultados_EDA/         # Gráficas de análisis exploratorio (CheXpert & Indiana)
│   └── Resultados_Evaluacion/  # Matrices, curvas ROC y métricas (Módulo 1, 2 y 3)
├── checkpoints/     # [Drive] Tensores de pesos (.pth) y adaptadores LoRA
├── data/            # [Drive] Espacio SSD para datasets clínicos estructurados
├── outputs/         # [Drive] Logs brutos
└── README.md        # Documentación arquitectónica