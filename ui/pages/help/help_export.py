"""
Página de ajuda sobre exportação de resultados do BioCine
"""

import streamlit as st
import pandas as pd


def show_export_help():
    """
    Renderiza a página de ajuda sobre exportação de resultados
    """
    st.markdown("<h1 class='main-header'>Exportação de Resultados</h1>", unsafe_allow_html=True)

    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>Compartilhando e Preservando seus Resultados</h3>
        <p>O BioCine oferece múltiplas opções para exportar seus resultados, facilitando o compartilhamento, a documentação e a análise posterior dos modelos cinéticos e previsões de machine learning.</p>
        </div>
    """, unsafe_allow_html=True)

    # Visão geral
    st.markdown("### Formatos de Exportação Disponíveis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### HTML")
        st.markdown("""
            **Descrição:** Relatório interativo e visual em formato web

            **Melhor para:**
            - Apresentações
            - Documentação completa
            - Compartilhamento de resultados visuais
            - Inclusão em relatórios técnicos
        """)

    with col2:
        st.markdown("#### Excel")
        st.markdown("""
            **Descrição:** Planilha estruturada com múltiplas abas

            **Melhor para:**
            - Análise posterior em planilhas
            - Manipulação e filtragem de dados
            - Criação de gráficos personalizados
            - Integração com outras análises
        """)

    with col3:
        st.markdown("#### CSV")
        st.markdown("""
            **Descrição:** Formato texto simples com valores separados por vírgula

            **Melhor para:**
            - Importação em outros softwares
            - Análises estatísticas avançadas
            - Compartilhamento de dados brutos
            - Armazenamento de longo prazo
        """)

    # Detalhamento de cada formato
    st.markdown("### Detalhamento dos Formatos")

    # HTML
    st.markdown("#### 1. Exportação para HTML")

    st.markdown("""
        O formato HTML gera um relatório completo que pode ser visualizado em qualquer navegador web. Este relatório inclui:

        - Resumo dos dados experimentais
        - Parâmetros e métricas dos modelos cinéticos
        - Resultados dos modelos de machine learning
        - Gráficos e visualizações interativas
        - Conclusões e observações

        **Opções de exportação:**
        - **ZIP com HTML + imagens:** Contém o relatório HTML e todas as imagens em alta resolução
        - **Apenas HTML:** Contém apenas o arquivo HTML (sem figuras)

        **Como exportar:**
        1. Acesse a página "Resultados"
        2. Selecione a guia "Exportação"
        3. Escolha "HTML" como formato de exportação
        4. Selecione as opções de conteúdo desejadas
        5. Clique em "Gerar Relatório HTML"
        6. Baixe o arquivo ZIP ou HTML

        **Dica:** O formato ZIP é recomendado para preservar a qualidade das figuras e garantir que o relatório seja visualizado corretamente em qualquer computador.
    """)

    # Exemplo de estrutura do relatório HTML
    st.markdown("**Estrutura do Relatório HTML:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            1. **Cabeçalho**
               - Título do relatório
               - Data de geração
               - Informações do projeto

            2. **Resumo dos Dados**
               - Número de amostras
               - Colunas disponíveis
               - Estatísticas descritivas

            3. **Modelos Cinéticos**
               - Parâmetros de cada modelo
               - Métricas de qualidade do ajuste
               - Gráficos de ajuste
        """)

    with col2:
        st.markdown("""
            4. **Machine Learning**
               - Características utilizadas
               - Métricas de desempenho
               - Importância das características
               - Gráficos de previsão

            5. **Figuras**
               - Gráficos de alta resolução
               - Visualizações interativas

            6. **Conclusões**
               - Resumo dos resultados
               - Observações relevantes
        """)

    # Excel
    st.markdown("#### 2. Exportação para Excel")

    st.markdown("""
        O formato Excel organiza os resultados em uma planilha estruturada com múltiplas abas, facilitando a análise posterior e manipulação dos dados. A planilha inclui:

        **Abas da planilha:**
        - **Dados Experimentais:** Dados originais importados ou gerados
        - **Previsões dos Modelos:** Valores previstos por cada modelo cinético
        - **Parâmetros:** Parâmetros ajustados de cada modelo
        - **Métricas:** Métricas de qualidade do ajuste (R², MSE)
        - **ML_[Modelo]_Métricas:** Métricas de desempenho do modelo de machine learning
        - **ML_[Modelo]_Importância:** Importância das características no modelo ML

        **Como exportar:**
        1. Acesse a página "Resultados"
        2. Selecione a guia "Exportação"
        3. Escolha "Excel" como formato de exportação
        4. Clique em "Gerar Excel"
        5. Baixe o arquivo Excel

        **Dica:** O Excel é ideal para quem deseja realizar análises adicionais ou criar gráficos personalizados a partir dos resultados.
    """)

    # Exemplo de estrutura da planilha Excel
    st.markdown("**Exemplo de Estrutura da Planilha Excel:**")

    excel_example = pd.DataFrame({
        'Aba': ['Dados Experimentais', 'Previsão_Logístico', 'Previsão_Monod', 'Parâmetros', 'Métricas',
                'ML_RandomForest_Métricas', 'ML_RandomForest_Importância'],
        'Conteúdo': [
            'Dados brutos originais (tempo, biomassa, substrato, etc.)',
            'Previsões do modelo Logístico',
            'Previsões do modelo de Monod (biomassa e substrato)',
            'Parâmetros ajustados de cada modelo (umax, ks, y, etc.)',
            'Métricas de qualidade do ajuste (R², MSE)',
            'Métricas de desempenho do modelo Random Forest (R² treino/teste, MSE treino/teste)',
            'Importância de cada característica no modelo Random Forest'
        ]
    })

    st.dataframe(excel_example)

    # CSV
    st.markdown("#### 3. Exportação para CSV")

    st.markdown("""
        O formato CSV (Comma-Separated Values) é um formato de texto simples e universal que pode ser importado por praticamente qualquer software de análise de dados. O BioCine permite exportar diferentes conjuntos de dados em CSV:

        **Opções de exportação CSV:**
        - **Dados Experimentais:** Dados brutos originais
        - **Previsões dos Modelos Cinéticos:** Valores previstos por cada modelo
        - **Métricas dos Modelos:** Métricas de qualidade do ajuste e parâmetros

        **Como exportar:**
        1. Acesse a página "Resultados"
        2. Selecione a guia "Exportação"
        3. Escolha "CSV" como formato de exportação
        4. Selecione o conteúdo específico para exportar
        5. Clique em "Gerar CSV"
        6. Baixe o arquivo CSV

        **Dica:** O formato CSV é ideal para importação em outros softwares estatísticos como R, Python (pandas), MATLAB, ou para preservação de longo prazo dos dados em um formato aberto.
    """)

    # Comparação de formatos
    st.markdown("### Comparação dos Formatos de Exportação")

    comparison = pd.DataFrame({
        'Característica': [
            'Completo (inclui gráficos)',
            'Interativo',
            'Editável',
            'Fácil importação em outros softwares',
            'Auto-contido (não requer software específico)',
            'Adequado para apresentações',
            'Adequado para análise posterior',
            'Adequado para armazenamento de longo prazo',
            'Tamanho do arquivo'
        ],
        'HTML': [
            '✅',
            '✅ (parcialmente)',
            '❌',
            '❌',
            '✅',
            '✅',
            '❌',
            '✅',
            'Grande'
        ],
        'Excel': [
            '❌ (sem gráficos)',
            '✅ (via Excel)',
            '✅',
            '✅ (softwares de planilha)',
            '❌ (requer Excel ou similar)',
            '❌',
            '✅',
            '❌ (formato proprietário)',
            'Médio'
        ],
        'CSV': [
            '❌ (sem gráficos)',
            '❌',
            '✅',
            '✅ (universal)',
            '✅',
            '❌',
            '✅',
            '✅',
            'Pequeno'
        ]
    })

    st.dataframe(comparison)

    # Recomendações
    st.markdown("### Recomendações de uso")

    st.markdown("""
        **Para relatórios completos e apresentações:**
        - Use a exportação HTML (ZIP) para preservar todas as visualizações
        - Ideal para compartilhar com colegas, incluir em relatórios ou apresentar resultados

        **Para análise posterior dos dados:**
        - Use a exportação Excel para manter a estrutura e possibilitar análises adicionais
        - Ideal para quem deseja manipular os dados, criar novos gráficos ou realizar cálculos adicionais

        **Para integração com outros softwares:**
        - Use a exportação CSV para máxima compatibilidade
        - Ideal para importação em R, Python, MATLAB ou outros softwares estatísticos

        **Para documentação completa do projeto:**
        - Exporte nos três formatos para garantir preservação e flexibilidade:
          - HTML para documentação visual completa
          - Excel para análises posteriores
          - CSV para preservação de longo prazo e compatibilidade máxima
    """)

    # Dicas e solução de problemas
    st.markdown("### Dicas e Solução de Problemas")

    st.markdown("""
        **Problemas com figuras no HTML:**
        - Se as figuras não aparecerem no relatório HTML, baixe a versão ZIP completa
        - Ao abrir o HTML do ZIP, mantenha a estrutura de pastas original

        **Problemas com Excel:**
        - Se houver erros ao abrir o Excel, verifique se você tem a versão mais recente do Microsoft Excel ou LibreOffice Calc
        - Alguns caracteres especiais podem não ser exibidos corretamente em versões mais antigas

        **Problemas com CSV:**
        - Ao importar CSV em outras ferramentas, verifique o delimitador (vírgula ou ponto-e-vírgula)
        - Alguns softwares podem interpretar incorretamente o separador decimal (ponto ou vírgula)

        **Dicas gerais:**
        - Para compartilhamento por e-mail, o formato ZIP (HTML) ou Excel é mais adequado
        - Para publicações científicas, exporte gráficos específicos em alta resolução
        - Mantenha os arquivos CSV originais para garantir reprodutibilidade
        - Inclua a data no nome do arquivo ao salvar múltiplas versões
    """)

    # Salvando, organizando e compartilhando resultados
    st.markdown("### Organizando e Compartilhando Resultados")

    st.markdown("""
        **Estrutura de pastas recomendada:**
        ```
        Projeto_Tratamento_Soro/
        ├── Dados/
        │   ├── Dados_Brutos/
        │   │   └── experimento_original.csv
        │   └── Dados_Processados/
        │       └── experimento_processado.csv
        ├── Modelos/
        │   ├── modelo_logistico_20240314.xlsx
        │   └── modelo_monod_20240314.xlsx
        ├── Relatórios/
        │   ├── relatorio_completo_20240314.zip
        │   └── resultados_resumidos_20240314.xlsx
        └── Publicação/
            └── figuras_alta_resolucao/
        ```

        **Nomenclatura de arquivos:**
        - Inclua sempre a data no formato AAAAMMDD
        - Adicione informações sobre o conteúdo (modelo, organismo, condições)
        - Exemplos:
          - `relatorio_chlorella_20240314.zip`
          - `modelagem_monod_fungos_20240314.xlsx`
          - `otimizacao_parametros_20240314.csv`

        **Compartilhamento de resultados:**
        - Para colaboradores: Compartilhe os arquivos Excel ou ZIP completos
        - Para publicações: Exporte figuras específicas em alta resolução
        - Para apresentações: Use o relatório HTML ou exporte slides com as visualizações principais
        - Para repositórios de dados: Inclua os arquivos CSV com metadados detalhados
    """)

    # Exemplos de uso
    st.markdown("### Exemplos de Uso dos Dados Exportados")

    tab1, tab2, tab3 = st.tabs(["Publicação Científica", "Relatório Técnico", "Análise Avançada"])

    with tab1:
        st.markdown("""
            **Preparando dados para uma publicação científica:**

            1. Execute a modelagem cinética completa no BioCine
            2. Exporte o relatório HTML para visualização completa
            3. Para figuras de publicação:
               - Identifique os gráficos mais relevantes
               - Use as visualizações avançadas para personalizar os gráficos
               - Exporte figuras individuais em alta resolução (PNG ou SVG)
            4. Para tabelas de publicação:
               - Exporte os parâmetros e métricas em Excel
               - Organize os dados conforme necessário para a publicação
               - Formate conforme as diretrizes da revista
            5. Para materiais suplementares:
               - Inclua os arquivos CSV completos
               - Adicione detalhes metodológicos da modelagem

            **Resultado:** Figuras profissionais, tabelas precisas e dados reproduzíveis para sua publicação.
        """)

    with tab2:
        st.markdown("""
            **Preparando um relatório técnico:**

            1. Execute a modelagem cinética e análise de ML completa
            2. Exporte o relatório HTML completo (ZIP)
            3. Incorpore o relatório ou suas seções em seu documento técnico
            4. Para apresentar aos stakeholders:
               - Destaque os gráficos de eficiência de remoção
               - Inclua a tabela de parâmetros otimizados
               - Apresente as condições ótimas de operação
            5. Para documentação técnica:
               - Inclua os arquivos Excel com todos os parâmetros
               - Documente as configurações utilizadas
               - Adicione notas sobre a interpretação dos resultados

            **Resultado:** Relatório técnico completo com visualizações profissionais e dados detalhados para tomada de decisão.
        """)

    with tab3:
        st.markdown("""
            **Realizando análises avançadas complementares:**

            1. Execute a modelagem no BioCine
            2. Exporte os resultados em formato CSV
            3. Importe os dados em software especializado:
               - **R:** Para análises estatísticas adicionais
                 ```r
                 dados <- read.csv("previsoes_modelos.csv")
                 library(ggplot2)
                 # Análise estatística avançada
                 ```
               - **Python:** Para integração com outros modelos
                 ```python
                 import pandas as pd
                 import seaborn as sns
                 dados = pd.read_csv("previsoes_modelos.csv")
                 # Análise adicional com scikit-learn
                 ```
               - **MATLAB:** Para modelagem mais avançada
                 ```matlab
                 dados = readtable('parametros_modelos.csv');
                 % Análise de sensibilidade avançada
                 ```
            4. Combine os resultados do BioCine com:
               - Análise de sensibilidade
               - Modelagem econômica
               - Análise de ciclo de vida
               - Simulação de escala industrial

            **Resultado:** Análise holística combinando modelagem cinética, aprendizado de máquina e outras abordagens analíticas.
        """)

    # Referências
    st.markdown("### Referências")

    st.markdown("""
        1. Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis. Springer-Verlag New York.

        2. McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56.

        3. Bengtsson, H. (2021). R.utils: Various Programming Utilities. R package version 2.10.1.

        4. Soares, A.P.M.R., Carvalho, F.O., De Farias Silva, C.E. (2020). Random Forest as a promising application to predict basic-dye biosorption process using orange waste. Journal of Environmental Chemical Engineering, v. 8, n. 4, 103952.
    """)