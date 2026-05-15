# ==========================================
# PREVISÃO DE NOTAS COM SCIKIT-LEARN
# ==========================================

# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression

# ==========================================
# BASE DE DADOS
# ==========================================

# Dados fornecidos
estudos = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# ==========================================
# PREPARAÇÃO DOS DADOS
# ==========================================

# Variável independente (X)
X = estudos[['horas']]

# Variável dependente (y)
y = estudos['notas']

# ==========================================
# CRIAÇÃO DO MODELO
# ==========================================

# Instanciando o modelo
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X, y)

# ==========================================
# INTERFACE STREAMLIT
# ==========================================

st.title("📚 Previsão de Nota por Horas de Estudo")

st.write("Modelo de Regressão Linear utilizando Scikit-Learn.")

# Entrada do usuário
horas_estudo = st.slider(
    "Selecione a quantidade de horas estudadas:",
    min_value=0,
    max_value=15,
    value=5
)

# ==========================================
# PREVISÃO
# ==========================================

# Realizando previsão
previsao = modelo.predict([[horas_estudo]])

# Exibindo resultado
st.subheader("Resultado da Previsão")

st.write(
    f"Para {horas_estudo} horas de estudo, "
    f"a nota prevista é aproximadamente:"
)

st.success(f"{previsao[0]:.2f}")

# ==========================================
# REPRESENTAÇÃO GRÁFICA
# ==========================================

# Criando gráfico
fig, ax = plt.subplots(figsize=(8, 5))

# Dados reais
ax.scatter(
    estudos['horas'],
    estudos['notas'],
    label='Dados Reais'
)

# Linha de regressão
ax.plot(
    estudos['horas'],
    modelo.predict(X),
    label='Linha de Regressão'
)

# Ponto previsto
ax.scatter(
    horas_estudo,
    previsao[0],
    s=120,
    label='Previsão Atual'
)

# Configurações do gráfico
ax.set_title('Regressão Linear - Horas vs Nota')
ax.set_xlabel('Horas de Estudo')
ax.set_ylabel('Nota')
ax.legend()

# Exibir gráfico no Streamlit
st.pyplot(fig)

# ==========================================
# INFORMAÇÕES DO MODELO
# ==========================================

st.subheader("Informações do Modelo")

st.write(f"Coeficiente Angular: {modelo.coef_[0]:.2f}")
st.write(f"Intercepto: {modelo.intercept_:.2f}")
