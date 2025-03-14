"""
Fornece ajuda contextual para diferentes partes da aplicação
"""

import streamlit as st


def show_context_help(context):
    """
    Mostra ajuda contextual específica para a parte da aplicação atual

    Args:
        context: String identificando o contexto (ex: 'monod', 'data_input', etc.)
    """
    st.markdown("<h2 class='sub-header'>Ajuda Contextual</h2>", unsafe_allow_html=True)

    if context == "data_input":
        show_data_input_help()
    elif context == "monod":
        show_monod_help()
    elif context == "logistic":
        show_logistic_help()
    elif context == "ml":
        show_ml_help()
    elif context == "parameters":
        show_parameters_help()
    elif context == "workflow":
        show_workflow_help()
    elif context == "export":
        show_export_help()
    elif context == "modeling":
        show_modeling_help()
    elif context == "results":
        show_results_help()
    elif context == "about":
        show_about_help()
    else:
        st.warning(f"Ajuda contextual não disponível para: {context}")


def show_data_input_help():
    """Ajuda contextual para a página de entrada de dados"""

    st.markdown("""
        ### Ajuda: Entrada de Dados

        Esta página permite importar ou gerar dados para análise no BioCine.

        **Importando CSV:**
        - O arquivo CSV deve conter dados experimentais do processo de tratamento
        - Colunas necessárias: tempo, biomassa
        - Colunas recomendadas: substrato, temperatura, pH, concentrações de poluentes

        **Configurações de Importação:**
        - **Delimitador**: Caractere que separa as colunas (geralmente vírgula ou ponto-e-vírgula)
        - **Separador Decimal**: Ponto (.) ou vírgula (,) 
        - **Linha de Cabeçalho**: Indica se a primeira linha contém os nomes das colunas
        - **Codificação**: Formato de codificação do arquivo (geralmente utf-8)

        **Pré-processamento:**
        - **Preencher valores ausentes**: Substitui valores ausentes pela média da coluna
        - **Normalizar dados**: Escala os dados para o intervalo [0,1]

        **Gerando Dados de Exemplo:**
        - Útil para testar as funcionalidades do BioCine
        - Os dados são gerados com base nos modelos cinéticos
        - Ajuste o número de amostras e a semente aleatória para diferentes conjuntos

        **Visualizando Dados:**
        - Explore os dados importados ou gerados
        - Verifique estatísticas descritivas
        - Crie gráficos para análise preliminar
    """)


def show_monod_help():
    """Ajuda contextual para o modelo de Monod"""

    st.markdown("""
        ### Ajuda: Modelo de Monod

        O modelo de Monod descreve a relação entre a taxa específica de crescimento e a concentração de substrato limitante.

        **Equações:**
        - μ = μmax * S / (Ks + S)
        - dX/dt = μ * X
        - dS/dt = -1/Y * μ * X

        **Parâmetros:**
        - **μmax**: Taxa específica máxima de crescimento (dia⁻¹)
        - **Ks**: Constante de meia saturação (mg/L)
        - **Y**: Coeficiente de rendimento (g biomassa/g substrato)

        **Quando usar:**
        - Quando você tem dados de crescimento de biomassa E consumo de substrato
        - Quando o crescimento é claramente limitado por um substrato específico

        **Requisitos de dados:**
        - Coluna de tempo (dias)
        - Coluna de biomassa (g/L)
        - Coluna de substrato (mg/L)

        **Interpretação:**
        - Um valor menor de Ks indica maior afinidade pelo substrato
        - Um valor maior de Y indica conversão mais eficiente de substrato em biomassa
    """)


def show_logistic_help():
    """Ajuda contextual para o modelo Logístico"""

    st.markdown("""
        ### Ajuda: Modelo Logístico

        O modelo Logístico descreve o crescimento de biomassa em um ambiente com recursos limitados.

        **Equação:**
        - dX/dt = μmax * X * (1 - X/Xmax)

        **Solução analítica:**
        - X(t) = Xmax / (1 + (Xmax/X0 - 1) * e^(-μmax * t))

        **Parâmetros:**
        - **μmax**: Taxa específica máxima de crescimento (dia⁻¹)
        - **X0**: Concentração inicial de biomassa (g/L)
        - **Xmax**: Concentração máxima de biomassa (g/L)

        **Quando usar:**
        - Quando você tem apenas dados de crescimento de biomassa (sem substrato)
        - Quando o crescimento segue um padrão sigmoidal
        - Quando o fator limitante não é claramente definido

        **Requisitos de dados:**
        - Coluna de tempo (dias)
        - Coluna de biomassa (g/L)

        **Interpretação:**
        - Xmax representa a capacidade de suporte do ambiente
        - μmax determina a inclinação da curva na fase exponencial
    """)


def show_ml_help():
    """Ajuda contextual para a página de machine learning"""

    st.markdown("""
        ### Ajuda: Machine Learning

        Esta página permite treinar modelos de machine learning para prever parâmetros do processo e otimizar condições.

        **Previsão de Parâmetros:**
        - **Parâmetro alvo**: Variável que você deseja prever (ex: biomassa, eficiência de remoção)
        - **Características**: Variáveis de entrada para o modelo (ex: tempo, temperatura, pH)
        - **Configuração do modelo**: Ajuste dos hiperparâmetros do Random Forest

        **Parâmetros do Random Forest:**
        - **Número de Árvores**: Mais árvores geralmente melhoram a precisão, mas aumentam o tempo de processamento
        - **Profundidade Máxima**: Controla a complexidade do modelo (valores maiores podem levar a overfitting)
        - **Proporção de Teste**: Porcentagem dos dados usados para teste (recomendado: 20%)

        **Avaliação do Modelo:**
        - **R²**: Coeficiente de determinação (quanto mais próximo de 1, melhor)
        - **MSE**: Erro quadrático médio (quanto menor, melhor)
        - **Importância das características**: Indica quais variáveis têm maior impacto na previsão

        **Otimização:**
        - Determina as condições ótimas para maximizar um objetivo específico
        - Você pode otimizar para máxima produção de biomassa ou máxima remoção de poluentes
        - Defina os intervalos para cada parâmetro baseados em valores realisticamente viáveis
    """)


def show_parameters_help():
    """Ajuda contextual para configuração de parâmetros"""

    st.markdown("""
        ### Ajuda: Parâmetros de Configuração

        Os parâmetros controlam o comportamento dos modelos cinéticos e de machine learning no BioCine.

        **Parâmetros do Modelo de Monod:**
        - **μmax**: Taxa específica máxima de crescimento (dia⁻¹)
          - Faixa típica: 0.1-1.0 (microalgas), 0.05-0.5 (fungos)
        - **Ks**: Constante de meia saturação (mg/L)
          - Faixa típica: 5-50 mg/L
        - **Y**: Coeficiente de rendimento (g biomassa/g substrato)
          - Faixa típica: 0.1-0.7

        **Parâmetros do Modelo Logístico:**
        - **μmax**: Taxa específica máxima de crescimento (dia⁻¹)
          - Faixa típica: 0.1-1.0 (microalgas), 0.05-0.5 (fungos)
        - **X0**: Concentração inicial de biomassa (g/L)
          - Faixa típica: 0.01-0.5 g/L
        - **Xmax**: Concentração máxima de biomassa (g/L)
          - Faixa típica: 1.0-8.0 (microalgas), 3.0-15.0 (fungos)

        **Parâmetros do Random Forest:**
        - **n_estimators**: Número de árvores (100-200 recomendado)
        - **max_depth**: Profundidade máxima (8-12 recomendado)
        - **min_samples_split**: Número mínimo de amostras para dividir um nó
        - **min_samples_leaf**: Número mínimo de amostras em um nó folha
    """)


def show_workflow_help():
    """Ajuda contextual para fluxo de trabalho"""

    st.markdown("""
        ### Ajuda: Fluxo de Trabalho

        O fluxo de trabalho recomendado no BioCine segue estas etapas:

        1. **Entrada de Dados**
           - Importe dados experimentais ou gere dados de exemplo
           - Verifique a qualidade dos dados e faça pré-processamento se necessário

        2. **Modelagem Cinética**
           - Ajuste o modelo Logístico se tiver apenas dados de biomassa
           - Ajuste o modelo de Monod se tiver dados de biomassa e substrato
           - Compare os modelos e selecione o mais adequado

        3. **Machine Learning**
           - Treine modelos para prever parâmetros relevantes
           - Identifique os fatores mais importantes
           - Determine condições ótimas através da otimização

        4. **Resultados**
           - Visualize e compare os resultados de diferentes modelos
           - Exporte relatórios ou dados para análise posterior

        Este fluxo não é estritamente linear - você pode retornar a etapas anteriores para refinar sua análise.
    """)


def show_export_help():
    """Ajuda contextual para exportação de resultados"""

    st.markdown("""
        ### Ajuda: Exportação de Resultados

        O BioCine oferece várias opções para exportar seus resultados:

        **Formatos disponíveis:**
        - **HTML**: Relatório completo com gráficos e tabelas
        - **Excel**: Planilha estruturada com múltiplas abas
        - **CSV**: Formato de texto simples para importação em outros softwares

        **O que é exportado:**
        - Dados experimentais originais
        - Parâmetros ajustados dos modelos
        - Previsões dos modelos cinéticos
        - Métricas de qualidade do ajuste
        - Resultados dos modelos de machine learning
        - Gráficos e visualizações (apenas no HTML)

        **Como exportar:**
        1. Acesse a página "Resultados"
        2. Selecione a guia "Exportação"
        3. Escolha o formato desejado
        4. Selecione as opções de conteúdo
        5. Clique no botão de exportação
        6. Baixe o arquivo gerado
    """)


def show_modeling_help():
    """Ajuda contextual para modelagem cinética"""

    st.markdown("""
        ### Ajuda: Modelagem Cinética

        Esta página permite ajustar modelos cinéticos aos seus dados experimentais:

        **Tipos de Modelos:**
        - **Logístico**: Modelo mais simples que requer apenas dados de biomassa
        - **Monod**: Modelo mais complexo que relaciona crescimento e consumo de substrato
        - **Comparar ambos**: Ajusta os dois modelos simultaneamente para comparação

        **Parâmetros dos Modelos:**
        - **Modelo Logístico**:
          - μmax: Taxa específica máxima de crescimento (dia⁻¹)
          - X0: Concentração inicial de biomassa (g/L)
          - Xmax: Concentração máxima de biomassa (g/L)

        - **Modelo de Monod**:
          - μmax: Taxa específica máxima de crescimento (dia⁻¹)
          - Ks: Constante de meia saturação (mg/L)
          - Y: Coeficiente de rendimento (g biomassa/g substrato)

        **Ajuste de Modelo:**
        1. Selecione o tipo de modelo
        2. Ajuste os parâmetros iniciais conforme necessário
        3. Clique em "Ajustar Modelo"
        4. Analise os resultados (curvas, parâmetros, métricas)

        **Simulação:**
        - Permite simular o comportamento do sistema com diferentes condições iniciais
        - Útil para prever o crescimento e consumo em situações não testadas experimentalmente
    """)


def show_results_help():
    """Ajuda contextual para página de resultados"""

    st.markdown("""
        ### Ajuda: Resultados

        Esta página permite visualizar e analisar os resultados dos modelos ajustados:

        **Resumo dos Resultados:**
        - Visão geral dos dados experimentais
        - Parâmetros e métricas dos modelos cinéticos
        - Métricas dos modelos de machine learning

        **Visualização Avançada:**
        - **Comparação de Modelos Cinéticos**: Compara as previsões de diferentes modelos
        - **Análise de Resíduos**: Avalia a qualidade do ajuste e identifica padrões nos erros
        - **Importância de Características**: Mostra quais variáveis têm maior impacto nas previsões de ML

        **Exportação:**
        - Exporta resultados em diferentes formatos (HTML, Excel, CSV)
        - Para relatórios completos, use o formato HTML
        - Para análise posterior, use Excel ou CSV

        **Dicas:**
        - Compare modelos diferentes para escolher o que melhor descreve seu sistema
        - Verifique se os resíduos seguem uma distribuição normal (indicativo de bom ajuste)
        - Use a análise de importância para identificar os fatores mais relevantes no processo
    """)


def show_about_help():
    """Ajuda contextual para página Sobre"""

    st.markdown("""
        ### Ajuda: Sobre o BioCine

        A página Sobre contém informações importantes sobre o software BioCine:

        **Fundamentação Teórica:**
        - Descrição dos modelos cinéticos implementados
        - Explicação das técnicas de machine learning utilizadas
        - Base científica para o tratamento de soro de leite

        **Informações do Software:**
        - Versão e data de lançamento
        - Licença de uso
        - Tecnologias utilizadas no desenvolvimento

        **Equipe de Desenvolvimento:**
        - Autores e colaboradores
        - Instituição de pesquisa
        - Projeto de pesquisa relacionado

        **Referências Bibliográficas:**
        - Artigos científicos que fundamentam a metodologia
        - Livros e outros materiais relevantes
        - Literatura sobre tratamento de soro de leite com microalgas e fungos

        **Contato:**
        - Informações para contato com a equipe de desenvolvimento
        - Links para website e repositório do projeto
    """)