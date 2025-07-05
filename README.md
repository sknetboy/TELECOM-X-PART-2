# Telecom X - Predicción de Cancelación (Churn)

![Telecom Industry](https://images.unsplash.com/photo-1551836022-d5d88e9218df?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2000&q=80)

## 📖 Descripción del Proyecto

Este proyecto aborda el desafío crítico de la cancelación de clientes (churn) en el sector telecomunicaciones. Como Analista Junior de Machine Learning en Telecom X, desarrollé modelos predictivos para identificar clientes con alta probabilidad de cancelar sus servicios, permitiendo a la empresa implementar estrategias proactivas de retención.

**Objetivo principal**: Construir un pipeline robusto de machine learning que prediga la cancelación de clientes con alta precisión e identifique los factores clave que influyen en esta decisión.

## 🔍 Hallazgos Clave

### 📈 Factores que Influyen en el Churn
| Factor | Impacto | Recomendación |
|--------|---------|---------------|
| **Tipo de contrato** | 3x mayor riesgo en contratos mensuales | Incentivar contratos anuales |
| **Tiempo como cliente** | 60% más churn en primeros 6 meses | Programa de fidelización |
| **Servicios adicionales** | 2.5x menos churn con seguridad online | Paquetes premium |
| **Forma de pago** | 40% más churn con factura electrónica | Descuentos por pago automático |
| **Gasto mensual** | Mayor churn en extremos (bajo/alto) | Personalizar planes |

### 🚀 Desempeño de Modelos
| Modelo | Exactitud | Precisión | Recall | F1-Score |
|--------|-----------|-----------|--------|----------|
| **Regresión Logística** | 0.82 | 0.82 | 0.82 | 0.82 |
| **Random Forest** | 0.85 | 0.85 | 0.85 | 0.85 |

## 🧠 Análisis Exploratorio

### Distribución de Churn
![Distribución de Churn](churn_distribution_before.png)

### Top Correlaciones con Churn
![Correlaciones con Churn](top_correlations.png)

### Tiempo como Cliente vs Churn
![Tiempo como Cliente](tenure_vs_churn.png)

### Gasto Total vs Churn
![Gasto Total](total_charges_vs_churn.png)

## ⚙️ Metodología

### 🔄 Pipeline de Machine Learning
1. **Preparación de datos**
   - Carga y limpieza de datos
   - Encoding de variables categóricas
   - Manejo de valores faltantes
   - Balanceo de clases con SMOTE

2. **Análisis exploratorio**
   - Matriz de correlaciones
   - Visualización de relaciones clave
   - Selección de características

3. **Modelado predictivo**
   - Entrenamiento de modelos (Regresión Logística y Random Forest)
   - Evaluación con múltiples métricas
   - Validación cruzada

4. **Interpretación de resultados**
   - Importancia de variables
   - Análisis de coeficientes
   - Matrices de confusión

### Matrices de Confusión
| Regresión Logística | Random Forest |
|---------------------|---------------|
| ![Confusión RL](confusion_matrix_logistic.png) | ![Confusión RF](confusion_matrix_rf.png) |

### Importancia de Variables
| Regresión Logística | Random Forest |
|---------------------|---------------|
| ![Coeficientes RL](logistic_coefficients.png) | ![Importancia RF](rf_feature_importance.png) |

## 💡 Conclusiones Estratégicas

### Principales Factores de Cancelación
1. **Contratos mensuales** tienen 3x mayor probabilidad de churn
2. **Clientes nuevos** (primeros 6 meses) son 60% más propensos a cancelar
3. **Falta de servicios adicionales** (seguridad, backup) aumenta riesgo
4. **Facturación electrónica** asociada con 40% más churn
5. **Gastos extremos** (muy bajos o muy altos) muestran mayor cancelación

### Estrategias de Retención
1. **Programa de fidelización** con beneficios progresivos
2. **Descuentos de 10-15%** en contratos anuales/bianuales
3. **Paquetes premium** de servicios de seguridad
4. **5% descuento** por pagos automáticos
5. **Intervención proactiva** para clientes de alto riesgo

## 🛠️ Reproducción del Análisis

### Prerrequisitos
- Python 3.8+
- Bibliotecas: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

### Instrucciones
1. Clonar repositorio:
```bash
git clone https://github.com/sknetboy/TELECOM-X-PART-2
cd Project-Telecom X-Part-2
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar análisis completo:
```bash
python telecom_churn_analysis.py
```

4. Los resultados se guardarán en:
- Visualizaciones en `/figuras/`
- Métricas en `/resultados/`
- Modelos serializados en `/modelos/`

## 📊 Estructura de Archivos
```
telecom-churn-prediction/
├── data/
│   └── telecom_cleaned.csv       # Datos tratados
├── notebooks/
│   └── Telecom_X_part_2.ipynb  # Jupyter Notebook
├── src/
│   └── telecom_churn_analysis.py # Script de análisis
├── figuras/                      # Visualizaciones
├── resultados/                   # Métricas y reportes
├── modelos/                      # Modelos entrenados
├── requirements.txt              # Dependencias
└── README.md                     # Este archivo
```

## 🧰 Tecnologías Utilizadas
**Lenguajes:** Python  
**Machine Learning:** scikit-learn, imbalanced-learn  
**Procesamiento de Datos:** pandas, numpy  
**Visualización:** matplotlib, seaborn  
**Gestión de Entorno:** pip, virtualenv  

## 🤝 Contribución
Las contribuciones son bienvenidas. Por favor:
1. Haz fork al proyecto
2. Crea tu rama (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios
4. Haz commit (`git commit -m 'Añade nueva funcionalidad'`)
5. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
6. Abre un Pull Request

## 📄 Licencia
Este proyecto está bajo la licencia [MIT](LICENSE).

---
**Desarrollado con ❤️ para Telecom X**  
*Analytics Team - 2025*
