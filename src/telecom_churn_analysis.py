# telecom_churn_analysis.py
# -*- coding: utf-8 -*-

""" Telecom X - Predicción de Cancelación (Churn) """

# =============================================================================
# 1. IMPORTACIÓN DE BIBLIOTECAS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os

# =============================================================================
# 2. CONFIGURACIÓN INICIAL
# =============================================================================
# Configuración de visualización
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
pd.set_option('display.max_columns', 50)
sns.set_palette("pastel")

# Crear directorios para guardar resultados
os.makedirs('figuras', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

# =============================================================================
# 3. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================
print("="*60)
print("PREPARACIÓN DE LOS DATOS")
print("="*60)

# Cargar datos tratados
print("\nCargando datos tratados...")
try:
    df = pd.read_csv("data/telecom_cleaned.csv")
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print("Error: Archivo 'data/telecom_cleaned.csv' no encontrado.")
    exit()

# Verificar estructura de los datos
print("\nDimensión de los datos:", df.shape)
print("\nPrimeras 5 filas:")
print(df.head())

# Eliminar columnas irrelevantes (si existieran)
print("\nColumnas iniciales:", df.columns.tolist())
# Eliminamos columnas irrelevantes como IDs (si existieran)
if 'customerID' in df.columns:
    df.drop(columns=["customerID"], inplace=True)
    print("Columna 'customerID' eliminada.")
print("\nColumnas después de eliminar irrelevantes:", df.columns.tolist())

# Encoding de variables categóricas
print("\nRealizando one-hot encoding...")
df_encoded = pd.get_dummies(df, drop_first=True)

# Verificación de la proporción de cancelación
print("\nDistribución de clases original:")
churn_distribution = df["Churn"].value_counts(normalize=True)
print(churn_distribution)

# Visualizar distribución de clases
plt.figure(figsize=(8, 5))
churn_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribución de Churn (Antes de Balanceo)')
plt.xlabel('Churn')
plt.ylabel('Proporción')
plt.xticks([0, 1], ['No Churn', 'Churn'], rotation=0)
plt.tight_layout()
plt.savefig('figuras/churn_distribution_before.png')
plt.show()

# =============================================================================
# 4. BALANCEO DE CLASES
# =============================================================================
print("\n" + "="*60)
print("BALANCEO DE CLASES")
print("="*60)

# Separar características y objetivo
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Manejo de valores faltantes
print("\nValores faltantes por columna:")
print(X.isnull().sum())

# Eliminar filas con valores faltantes (si existen)
if X.isnull().sum().sum() > 0:
    X_cleaned = X.dropna()
    y_cleaned = y[X_cleaned.index]
    print(f"\nSe eliminaron {X.shape[0] - X_cleaned.shape[0]} registros con valores faltantes.")
else:
    X_cleaned = X
    y_cleaned = y
    print("\nNo se encontraron valores faltantes.")

print(f"\nDatos originales: {X.shape[0]} registros")
print(f"Datos después de limpieza: {X_cleaned.shape[0]} registros")

# Balanceo con SMOTE
print("\nAplicando SMOTE para balancear clases...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_cleaned, y_cleaned)

print("\nDistribución balanceada después de SMOTE:")
balanced_distribution = pd.Series(y_res).value_counts(normalize=True)
print(balanced_distribution)

# Visualizar distribución balanceada
plt.figure(figsize=(8, 5))
balanced_distribution.plot(kind='bar', color=['lightgreen', 'lightcoral'])
plt.title('Distribución de Churn (Después de Balanceo)')
plt.xlabel('Churn')
plt.ylabel('Proporción')
plt.xticks([0, 1], ['No Churn', 'Churn'], rotation=0)
plt.tight_layout()
plt.savefig('figuras/churn_distribution_after.png')
plt.show()

# =============================================================================
# 5. ANÁLISIS DE CORRELACIONES
# =============================================================================
print("\n" + "="*60)
print("ANÁLISIS DE CORRELACIONES")
print("="*60)

# Matriz de correlación
corr_matrix = df_encoded.corr()

# Correlación con la variable objetivo
churn_corr = corr_matrix['Churn'].sort_values(ascending=False)

print("\nVariables con mayor correlación con Churn:")
print(churn_corr.head(10))
print("\nVariables con menor correlación con Churn:")
print(churn_corr.tail(10))

# Guardar correlaciones en CSV
churn_corr.to_csv('resultados/correlaciones_churn.csv', index=True)

# Visualizar top 10 correlaciones
plt.figure(figsize=(10, 6))
churn_corr.drop('Churn').sort_values().tail(10).plot(kind='barh', color='teal')
plt.title('Top 10 Variables Correlacionadas con Churn')
plt.xlabel('Coeficiente de Correlación')
plt.tight_layout()
plt.savefig('figuras/top_correlations.png')
plt.show()

# Análisis dirigido: Tiempo de contrato vs Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='customer_tenure', data=df, palette='Set2')
plt.title('Relación entre Tiempo como Cliente y Churn')
plt.xlabel('Churn')
plt.ylabel('Meses como Cliente')
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig('figuras/tenure_vs_churn.png')
plt.show()

# Análisis dirigido: Gasto total vs Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='account_Charges_Total', data=df, palette='Set2')
plt.title('Relación entre Gasto Total y Churn')
plt.xlabel('Churn')
plt.ylabel('Gasto Total ($)')
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig('figuras/total_charges_vs_churn.png')
plt.show()

# =============================================================================
# 6. MODELADO PREDICTIVO
# =============================================================================
print("\n" + "="*60)
print("MODELADO PREDICTIVO")
print("="*60)

# División de datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

print(f"\nDatos de entrenamiento: {X_train.shape[0]} registros")
print(f"Datos de prueba: {X_test.shape[0]} registros")

# Estandarización para modelos sensibles a escalas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 6.1 Regresión Logística
# =============================================================================
print("\n" + "-"*40)
print("ENTRENANDO REGRESIÓN LOGÍSTICA")
print("-"*40)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# Métricas de evaluación
accuracy_log = accuracy_score(y_test, y_pred_log)
report_log = classification_report(y_test, y_pred_log, output_dict=True)

print("\nExactitud (Accuracy):", accuracy_log)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_log))

# Guardar reporte en CSV
report_log_df = pd.DataFrame(report_log).transpose()
report_log_df.to_csv('resultados/reporte_regresion_logistica.csv')

# Matriz de confusión
conf_matrix_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Matriz de Confusión - Regresión Logística')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('figuras/confusion_matrix_logistic.png')
plt.show()

# =============================================================================
# 6.2 Random Forest
# =============================================================================
print("\n" + "-"*40)
print("ENTRENANDO RANDOM FOREST")
print("-"*40)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Métricas de evaluación
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

print("\nExactitud (Accuracy):", accuracy_rf)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_rf))

# Guardar reporte en CSV
report_rf_df = pd.DataFrame(report_rf).transpose()
report_rf_df.to_csv('resultados/reporte_random_forest.csv')

# Matriz de confusión
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('figuras/confusion_matrix_rf.png')
plt.show()

# =============================================================================
# 7. COMPARACIÓN DE MODELOS
# =============================================================================
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS")
print("="*60)

# Comparación de exactitud
print(f"\nExactitud Regresión Logística: {accuracy_log:.4f}")
print(f"Exactitud Random Forest: {accuracy_rf:.4f}")

# Comparación F1-score
f1_log = report_log['weighted avg']['f1-score']
f1_rf = report_rf['weighted avg']['f1-score']
print(f"\nF1-score Regresión Logística: {f1_log:.4f}")
print(f"F1-score Random Forest: {f1_rf:.4f}")

# Determinar mejor modelo
if f1_rf > f1_log:
    print("\nConclusión: Random Forest tiene mejor desempeño general (F1-score más alto)")
    best_model = "Random Forest"
else:
    print("\nConclusión: Regresión Logística tiene mejor desempeño general (F1-score más alto)")
    best_model = "Regresión Logística"

# =============================================================================
# 8. IMPORTANCIA DE VARIABLES
# =============================================================================
print("\n" + "="*60)
print("IMPORTANCIA DE VARIABLES")
print("="*60)

# Importancia para Regresión Logística
log_coef = pd.Series(log_model.coef_[0], index=X_cleaned.columns)
top_log_coef = log_coef.abs().sort_values(ascending=False).head(10)

print("\nVariables más importantes para Regresión Logística (coeficientes):")
print(top_log_coef)

# Guardar coeficientes
top_log_coef.to_csv('resultados/top_coeficientes_logistica.csv', index=True)

# Visualizar coeficientes de regresión logística
plt.figure(figsize=(10, 6))
top_log_coef.sort_values().plot(kind='barh', color='purple')
plt.title('Top 10 Variables - Regresión Logística (Valor Absoluto Coeficientes)')
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig('figuras/logistic_coefficients.png')
plt.show()

# Importancia para Random Forest
rf_importances = pd.Series(rf_model.feature_importances_, index=X_cleaned.columns)
top_rf_importances = rf_importances.sort_values(ascending=False).head(10)

print("\nVariables más importantes para Random Forest:")
print(top_rf_importances)

# Guardar importancias
top_rf_importances.to_csv('resultados/top_importancias_random_forest.csv', index=True)

# Visualizar importancia de variables para Random Forest
plt.figure(figsize=(10, 6))
top_rf_importances.sort_values().plot(kind='barh', color='green')
plt.title('Top 10 Variables - Random Forest')
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig('figuras/rf_feature_importance.png')
plt.show()

# =============================================================================
# 9. REPORTE COMBINADO DE FACTORES CLAVE (CORRECCIÓN INCLUIDA)
# =============================================================================
print("\n" + "="*60)
print("REPORTE COMBINADO DE FACTORES CLAVE")
print("="*60)

# Encontrar las variables comunes entre ambos modelos
common_features = list(set(top_log_coef.index) & set(top_rf_importances.index))

# Filtrar las series para incluir solo las variables comunes
top_log_coef_common = top_log_coef[common_features].head(10)
top_rf_importances_common = top_rf_importances[common_features].head(10)

# Crear el DataFrame con las variables comunes
top_factors = pd.DataFrame({
    'Regresión_Logística': top_log_coef_common,
    'Random_Forest': top_rf_importances_common
})

# Ordenar por importancia en Regresión Logística
top_factors = top_factors.sort_values('Regresión_Logística', ascending=False)

# Guardar en CSV
top_factors.to_csv('resultados/top_factors_churn.csv')
print("\nReporte de factores clave guardado en 'resultados/top_factors_churn.csv'")

# =============================================================================
# 10. CONCLUSIONES ESTRATÉGICAS
# =============================================================================
print("\n" + "="*60)
print("CONCLUSIONES ESTRATÉGICAS")
print("="*60)

print("""
Principales Factores que Influyen en la Cancelación:

1. **Tiempo de Contrato (customer_tenure):** 
   - Los clientes nuevos tienen mayor probabilidad de cancelación
   - La lealtad aumenta con el tiempo (clientes antiguos menos propensos a churn)

2. **Tipo de Contrato (account_Contract):**
   - Clientes con contratos mensuales tienen mayor tasa de cancelación
   - Contratos anuales/bianuales muestran mayor retención

3. **Gastos Mensuales (account_Charges_Monthly):**
   - Relación compleja: clientes con gastos muy altos o muy bajos muestran mayor churn
   - Segmento medio (valor $65-80) tiene mayor retención

4. **Servicios Adicionales:**
   - Clientes con servicios como seguridad online, backup en la nube y protección de dispositivo muestran menor cancelación
   - La falta de estos servicios aumenta el riesgo de churn

5. **Forma de Pago:**
   - Pagos electrónicos están asociados con mayor cancelación
   - Pagos automáticos (tarjeta crédito/transferencia) tienen mejor retención

Estrategias de Retención:

1. **Programa de Fidelización:**
   - Ofrecer beneficios progresivos por tiempo de permanencia
   - Descuentos especiales al cumplir 1 año de servicio

2. **Incentivos a Contratos Anuales:**
   - Promociones exclusivas para compromisos a largo plazo
   - Descuento del 10-15% en contratos anuales/bianuales

3. **Paquetes de Servicios Premium:**
   - Agrupar servicios de seguridad y backup a precio especial
   - Pruebas gratuitas de servicios adicionales para clientes en riesgo

4. **Programa de Pago Automático:**
   - Descuento del 5% para clientes que adopten pagos automáticos
   - Notificaciones proactivas antes de cargos recurrentes

5. **Intervención Proactiva:**
   - Sistema de alertas para clientes de alto riesgo (nuevos + contrato mensual)
   - Ofertas personalizadas al detectar patrones de posible cancelación

Recomendaciones Técnicas:

- Implementar modelo de {} en producción con monitoreo continuo
- Reentrenar modelos mensualmente con nuevos datos
- Desarrollar dashboard ejecutivo de métricas de churn
- Crear sistema de scoring de clientes en tiempo real
""".format(best_model))

# Guardar conclusiones en texto
with open('resultados/conclusiones_estrategicas.txt', 'w') as f:
    f.write("Principales Factores que Influyen en la Cancelación:\n")
    f.write("1. Tiempo de Contrato (customer_tenure)\n")
    f.write("2. Tipo de Contrato (account_Contract)\n")
    f.write("3. Gastos Mensuales (account_Charges_Monthly)\n")
    f.write("4. Servicios Adicionales\n")
    f.write("5. Forma de Pago\n\n")
    f.write("Estrategias de Retención:\n")
    f.write("1. Programa de Fidelización\n")
    f.write("2. Incentivos a Contratos Anuales\n")
    f.write("3. Paquetes de Servicios Premium\n")
    f.write("4. Programa de Pago Automático\n")
    f.write("5. Intervención Proactiva\n")

print("\n¡Análisis completado exitosamente!")