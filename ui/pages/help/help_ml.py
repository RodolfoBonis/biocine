"""
Página de ajuda sobre Machine Learning
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression


def show_ml_help():
    """
    Renderiza a página de ajuda sobre Machine Learning
    """
    st.markdown("<h1 class='main-header'>Machine Learning no BioCine</h1>", unsafe_allow_html=True)

    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>O que é Machine Learning no contexto do BioCine?</h3>
        <p>O BioCine utiliza algoritmos de Machine Learning para prever parâmetros de processo como crescimento de biomassa ou eficiência de remoção de poluentes, bem como para otimizar condições de operação do tratamento biológico.</p>
        </div>
    """, unsafe_allow_html=True)

    # Random Forest
    st.markdown("### Algoritmo Random Forest")

    st.markdown("""
        O principal algoritmo implementado no BioCine é o Random Forest, um método de ensemble learning baseado em árvores de decisão. Este algoritmo é utilizado para:

        1. **Predição de parâmetros de processo**: Crescimento de biomassa, remoção de poluentes (N, P, DQO)
        2. **Identificação de fatores importantes**: Determinação das variáveis mais relevantes para o processo
        3. **Otimização**: Encontrar as condições ideais para maximizar a eficiência do tratamento

        O Random Forest combina múltiplas árvores de decisão para produzir um modelo mais robusto e preciso, reduzindo o risco de overfitting.
    """)

    # Visualização interativa do Random Forest
    st.markdown("### Visualização do Funcionamento do Random Forest")

    # Configurações interativas
    st.markdown("Ajuste os parâmetros para ver como o modelo Random Forest se comporta:")

    col1, col2 = st.columns(2)

    with col1:
        n_estimators = st.slider("Número de Árvores", 10, 200, 100, 10)
        max_depth = st.slider("Profundidade Máxima", 1, 20, 10, 1)

    with col2:
        n_samples = st.slider("Tamanho do Conjunto de Dados", 50, 500, 200, 50)
        noise = st.slider("Nível de Ruído", 0.0, 1.0, 0.25, 0.05)

    # Gerar dados sintéticos para demonstração
    X, y = make_regression(n_samples=n_samples, n_features=4, noise=noise, random_state=42)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar modelo
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Fazer predições
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # Calcular métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Exibir métricas
    st.markdown("### Métricas de Desempenho do Modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("R² (Treino)", f"{r2_train:.4f}")
        st.metric("MSE (Treino)", f"{mse_train:.4f}")

    with col2:
        st.metric("R² (Teste)", f"{r2_test:.4f}")
        st.metric("MSE (Teste)", f"{mse_test:.4f}")

    # Visualizar importância das características
    st.markdown("### Importância das Características")

    # Gerar importância das características
    importance = rf.feature_importances_

    # Criar DataFrame para plotly
    importance_df = pd.DataFrame({
        'Feature': [f'Feature {i + 1}' for i in range(4)],
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Plotar com Plotly
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Importância das Características',
        text='Importance',
        labels={'Importance': 'Importância', 'Feature': 'Característica'}
    )

    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(xaxis_range=[0, importance_df['Importance'].max() * 1.1])

    st.plotly_chart(fig, use_container_width=True)

    # Visualizar predições vs valores reais
    st.markdown("### Predições vs Valores Reais")

    df_test = pd.DataFrame({
        'Valores Reais': y_test,
        'Predições': y_pred_test
    })

    fig = px.scatter(
        df_test,
        x='Valores Reais',
        y='Predições',
        title='Predições vs Valores Reais'
    )

    # Adicionar linha de referência
    min_val = min(df_test['Valores Reais'].min(), df_test['Predições'].min())
    max_val = max(df_test['Valores Reais'].max(), df_test['Predições'].max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='y=x',
            line=dict(color='red', dash='dash')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Parâmetros do Random Forest
    st.markdown("### Parâmetros do Random Forest")

    st.markdown("""
        | Parâmetro | Descrição | Efeito no Modelo | Quando Ajustar |
        |-----------|-----------|------------------|----------------|
        | `n_estimators` | Número de árvores na floresta | Mais árvores geralmente resultam em melhor desempenho, mas aumentam o tempo de processamento | Aumentar se o modelo está subajustado (underfitting); diminuir para reduzir tempo de processamento |
        | `max_depth` | Profundidade máxima de cada árvore | Controla a complexidade do modelo; valores maiores permitem capturar relações mais complexas | Diminuir se o modelo está superajustado (overfitting); aumentar se está subajustado |
        | `min_samples_split` | Número mínimo de amostras para dividir um nó | Ajuda a controlar o overfitting | Aumentar para reduzir overfitting em conjuntos pequenos |
        | `min_samples_leaf` | Número mínimo de amostras em um nó folha | Também ajuda a prevenir overfitting | Aumentar para tornar o modelo mais conservador |
        | `random_state` | Semente para geração de números aleatórios | Garante reprodutibilidade dos resultados | Manter um valor fixo (como 42) para resultados consistentes |
    """)

    # Processo de Machine Learning
    st.markdown("### Processo de Machine Learning no BioCine")

    st.markdown("""
        O processo de machine learning no BioCine segue estas etapas:

        1. **Preparação de dados**
           - Seleção das características relevantes (tempo, temperatura, pH, concentrações iniciais)
           - Seleção da variável alvo (crescimento de biomassa, remoção de poluentes)
           - Divisão em conjuntos de treino e teste (padrão: 80% treino, 20% teste)

        2. **Treinamento do modelo**
           - Configuração dos hiperparâmetros do Random Forest
           - Ajuste do modelo aos dados de treino
           - Avaliação da qualidade do ajuste

        3. **Análise de resultados**
           - Avaliação das métricas de desempenho (R², MSE)
           - Análise da importância das características
           - Visualização de predições vs valores reais

        4. **Aplicação do modelo**
           - Previsão de novos valores
           - Otimização de parâmetros do processo
           - Identificação de condições ideais
    """)

    # Aplicações e quando usar
    st.markdown("### Aplicações e Quando Usar")

    st.markdown("""
        O machine learning no BioCine é particularmente útil para:

        - **Predição da eficiência de remoção**: Estimar a capacidade de remoção de nitrogênio, fósforo e DQO com base em parâmetros iniciais
        - **Otimização do processo**: Determinar as condições ideais (temperatura, pH, tempo) para maximizar a eficiência do tratamento
        - **Compreensão do processo**: Identificar quais fatores têm maior impacto na eficiência do tratamento
        - **Redução de experimentos**: Minimizar a necessidade de experimentos adicionais ao prever o comportamento do sistema

        **Quando usar machine learning no BioCine:**

        - Quando você possui um conjunto de dados suficiente (pelo menos 15-20 pontos experimentais)
        - Quando as relações entre variáveis são complexas e não lineares
        - Quando deseja prever o comportamento do sistema em condições não testadas experimentalmente
        - Quando busca otimizar múltiplos parâmetros simultaneamente
    """)

    # Limitações
    st.markdown("### Limitações e Considerações")

    st.markdown("""
        Ao utilizar machine learning no BioCine, tenha em mente:

        - **Qualidade dos dados**: A precisão do modelo depende diretamente da qualidade e quantidade dos dados experimentais
        - **Extrapolação**: Modelos de ML podem não ser confiáveis ao extrapolar muito além dos dados de treinamento
        - **Interpretabilidade**: Embora o Random Forest forneça importância de características, a interpretação física direta pode ser desafiadora
        - **Validação**: Sempre valide os resultados do modelo com conhecimento do domínio e, quando possível, com experimentos adicionais

        Random Forest é menos propenso a overfitting que outros algoritmos, mas ainda requer validação cuidadosa, especialmente com conjuntos de dados pequenos.
    """)

    # Referências
    st.markdown("### Referências")

    st.markdown("""
        1. Soares, A.P.M.R., Carvalho, F.O., & De Farias Silva, C.E. (2020). Random Forest as a promising application to predict basic-dye biosorption process using orange waste. Journal of Environmental Chemical Engineering, v. 8, n. 4, 103952.

        2. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

        3. Prasad, A. M., Iverson, L. R., & Liaw, A. (2006). Newer classification and regression tree techniques: bagging and random forests for ecological prediction. Ecosystems, 9(2), 181-199.
    """)