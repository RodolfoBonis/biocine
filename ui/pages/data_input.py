"""
Página de entrada de dados do BioCine

Esta página permite a importação de dados experimentais ou
a geração de dados de exemplo para a modelagem cinética.
"""

import streamlit as st
import pandas as pd
import os
import numpy as np
from utils import import_csv, generate_example_data, validate_data, preprocess_data, save_processed_data
from ui.pages.help import show_context_help

def show_data_input():
    """
    Renderiza a página de entrada de dados
    """
    st.markdown("<h2 class='sub-header'>Entrada de Dados</h2>", unsafe_allow_html=True)

    # Botão de ajuda
    col1, col2 = st.columns([10, 1])
    with col2:
        help_button = st.button("ⓘ", help="Mostrar ajuda sobre entrada de dados")
        if help_button:
            st.session_state.show_help = True
            st.session_state.help_context = "data_input"
            return st.rerun()


    # Tabs para diferentes formas de entrada de dados
    tab1, tab2, tab3 = st.tabs(["Importar CSV", "Gerar Dados de Exemplo", "Visualizar Dados"])

    with tab1:
        st.subheader("Importar Dados de Arquivo CSV")

        # Upload de arquivo CSV
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

        if uploaded_file is not None:
            st.success("Arquivo carregado com sucesso!")

            # Opções de importação
            st.subheader("Opções de Importação")

            col1, col2 = st.columns(2)
            with col1:
                delimiter = st.selectbox("Delimitador", [",", ";", "\t"], index=0)
                decimal = st.selectbox("Separador Decimal", [".", ","], index=0)

            with col2:
                header = st.selectbox("Linha de Cabeçalho", ["Sim", "Não"], index=0)
                encoding = st.selectbox("Codificação", ["utf-8", "latin1", "iso-8859-1"], index=0)

            # Pré-visualização dos dados
            st.subheader("Pré-visualização dos Dados")

            try:
                # Carrega os dados
                data = import_csv(
                    uploaded_file,
                    delimiter=delimiter,
                    decimal=decimal,
                    header=0 if header == "Sim" else None,
                    encoding=encoding
                )

                # Mostra os primeiros registros
                st.dataframe(data.head(10))

                # Configuração de colunas
                st.subheader("Configuração de Colunas")

                # Verifica se há colunas adequadas para modelagem
                if validate_data(data):
                    # Permite ao usuário selecionar as colunas relevantes
                    time_col = st.selectbox("Coluna de Tempo", data.columns)
                    biomass_col = st.selectbox("Coluna de Biomassa", data.columns)
                    substrate_col = st.selectbox("Coluna de Substrato (opcional)",
                                               ["Nenhuma"] + list(data.columns))

                    # Opções de pré-processamento
                    st.subheader("Pré-processamento")

                    preprocessing_options = st.multiselect(
                        "Opções de Pré-processamento",
                        ["Preencher valores ausentes", "Normalizar dados"],
                        ["Preencher valores ausentes"]
                    )

                    # Botão para processar os dados
                    if st.button("Processar Dados"):
                        try:
                            # Pré-processamento
                            fill_missing = "Preencher valores ausentes" in preprocessing_options
                            normalize = "Normalizar dados" in preprocessing_options

                            processed_data = preprocess_data(data, fill_missing, normalize)

                            # Renomeia as colunas para padronização
                            column_mapping = {
                                time_col: "tempo",
                                biomass_col: "biomassa"
                            }

                            if substrate_col != "Nenhuma":
                                column_mapping[substrate_col] = "substrato"

                            processed_data = processed_data.rename(columns=column_mapping)

                            # Salva os dados no estado da sessão
                            st.session_state.data = processed_data

                            # Salva os dados em um arquivo na pasta processed
                            import os
                            import yaml
                            import datetime

                            # Carrega configurações para obter o caminho do diretório
                            config_path = os.path.join(
                                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "config", "config.yaml")

                            with open(config_path, "r") as f:
                                config = yaml.safe_load(f)

                            # Define o caminho do arquivo
                            processed_dir = os.path.join(
                                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                config['paths']['processed_data'])

                            os.makedirs(processed_dir, exist_ok=True)

                            # Cria um nome descritivo para o arquivo
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            source_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                            options_suffix = f"_fill{int(fill_missing)}_norm{int(normalize)}"

                            filename = f"{source_name}_processed_{timestamp}{options_suffix}.csv"
                            file_path = os.path.join(processed_dir, filename)

                            # Salva os dados
                            from utils.data_processor import save_processed_data
                            success = save_processed_data(processed_data, file_path)

                            # Mensagens de sucesso
                            st.success("Dados processados com sucesso!")

                            if success:
                                st.info(f"Dados processados salvos em: {file_path}")

                            st.info("Vá para a guia 'Visualizar Dados' para explorar os dados.")
                        except Exception as e:
                            st.error(f"Erro ao processar os dados: {str(e)}")
                else:
                    st.warning("Os dados importados não têm o formato adequado para modelagem cinética.")

            except Exception as e:
                st.error(f"Erro ao importar os dados: {str(e)}")

    with tab2:
        st.subheader("Gerar Dados de Exemplo")

        # Configurações para geração de dados
        st.markdown("Configure os parâmetros para gerar dados sintéticos de exemplo.")

        col1, col2 = st.columns(2)

        with col1:
            n_samples = st.slider("Número de Amostras", 10, 100, 20)
            seed = st.number_input("Semente Aleatória", 0, 1000, 42)

        # Botão para gerar dados
        if st.button("Gerar Dados de Exemplo"):
            try:
                # Gera os dados
                example_data = generate_example_data(n_samples, seed)

                # Salva os dados no estado da sessão
                st.session_state.data = example_data

                # Salva os dados em um arquivo
                import os
                import yaml

                # Carrega configurações para obter o caminho do diretório
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "config", "config.yaml")

                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                # Define o caminho do arquivo
                example_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           config['paths']['example_data'])

                os.makedirs(example_dir, exist_ok=True)

                filename = f"example_data_n{n_samples}_seed{seed}.csv"
                file_path = os.path.join(example_dir, filename)

                # Salva os dados
                from utils.data_processor import save_processed_data
                success = save_processed_data(example_data, file_path)

                # Mensagens de sucesso
                st.success("Dados de exemplo gerados com sucesso!")

                if success:
                    st.info(f"Dados salvos em: {file_path}")

                st.info("Vá para a guia 'Visualizar Dados' para explorar os dados.")

            except Exception as e:
                st.error(f"Erro ao gerar dados de exemplo: {str(e)}")

    with tab3:
        st.subheader("Visualizar Dados")

        if 'data' in st.session_state and st.session_state.data is not None:
            data = st.session_state.data

            # Informações básicas
            st.markdown("### Informações Básicas")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Número de Amostras:** {len(data)}")
                st.markdown(f"**Colunas:** {', '.join(data.columns)}")

            # Mostra os dados
            st.markdown("### Dados")
            st.dataframe(data)

            # Estatísticas descritivas
            st.markdown("### Estatísticas Descritivas")
            st.dataframe(data.describe())

            # Visualizações
            st.markdown("### Visualizações")

            # Seleciona colunas para visualizar
            if len(data.columns) > 1:
                viz_cols = st.multiselect(
                    "Selecione as colunas para visualizar",
                    data.select_dtypes(include=[np.number]).columns,
                    default=["tempo", "biomassa"] if all(col in data.columns for col in ["tempo", "biomassa"]) else data.select_dtypes(include=[np.number]).columns[:2]
                )

                if len(viz_cols) > 0:
                    # Tipo de gráfico
                    plot_type = st.selectbox(
                        "Tipo de Gráfico",
                        ["Linha", "Dispersão", "Histograma", "Boxplot", "Matriz de Correlação"]
                    )

                    if plot_type == "Linha":
                        # Usando st.line_chart para gráficos de linha interativos
                        if "tempo" in data.columns:
                            # Gráfico com tempo no eixo X
                            line_data = pd.DataFrame()
                            for col in viz_cols:
                                if col != "tempo":
                                    line_data[col] = data[col].values
                            if not line_data.empty:
                                line_data.index = data["tempo"].values
                                st.line_chart(line_data)
                        else:
                            # Gráfico com índices no eixo X
                            st.line_chart(data[viz_cols])

                    elif plot_type == "Dispersão":
                        if len(viz_cols) >= 2:
                            # Usando st.scatter_chart para gráficos de dispersão interativos
                            # Em versões mais antigas do Streamlit, pode ser necessário usar Altair ou Plotly
                            scatter_data = pd.DataFrame({
                                'x': data[viz_cols[0]],
                                'y': data[viz_cols[1]]
                            })

                            # Verifica se a versão do Streamlit suporta scatter_chart
                            try:
                                st.scatter_chart(
                                    scatter_data,
                                    x='x',
                                    y='y'
                                )
                            except AttributeError:
                                # Fallback para versões mais antigas usando plotly
                                import plotly.express as px
                                fig = px.scatter(scatter_data, x='x', y='y')
                                fig.update_layout(
                                    xaxis_title=viz_cols[0],
                                    yaxis_title=viz_cols[1]
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Selecione pelo menos duas colunas para o gráfico de dispersão.")

                    elif plot_type == "Histograma":
                        # Usando st.bar_chart para histogramas interativos
                        for col in viz_cols:
                            counts, bins = np.histogram(data[col], bins=10)
                            hist_data = pd.DataFrame({
                                'count': counts
                            }, index=[f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)])

                            st.subheader(f"Histograma de {col}")
                            st.bar_chart(hist_data)

                    elif plot_type == "Boxplot":
                        # Usando plotly para boxplots interativos
                        import plotly.express as px

                        boxplot_data = pd.melt(data[viz_cols], var_name='Variável', value_name='Valor')
                        fig = px.box(boxplot_data, x='Variável', y='Valor')
                        st.plotly_chart(fig, use_container_width=True)

                    elif plot_type == "Matriz de Correlação":
                        # Calculando a matriz de correlação
                        corr_matrix = data[viz_cols].corr()

                        # Usando plotly para heatmap interativo
                        import plotly.express as px

                        # Convertendo a matriz para formato long para o plotly
                        corr_data = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(len(corr_matrix.index)):
                                corr_data.append({
                                    'x': corr_matrix.columns[i],
                                    'y': corr_matrix.index[j],
                                    'correlation': corr_matrix.iloc[j, i]
                                })

                        corr_df = pd.DataFrame(corr_data)

                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            color_continuous_scale='RdBu_r',
                            range_color=[-1, 1]
                        )

                        st.plotly_chart(fig, use_container_width=True)

            # Opções de salvamento
            st.markdown("### Salvar Dados")

            save_format = st.selectbox("Formato de Salvamento", ["CSV", "Excel"])

            if st.button("Salvar Dados"):
                if save_format == "CSV":
                    # Salva como CSV
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="biocine_data.csv",
                        mime="text/csv"
                    )
                else:
                    # Salva como Excel
                    try:
                        temp_path = "temp_data.xlsx"
                        data.to_excel(temp_path, index=False)

                        with open(temp_path, "rb") as f:
                            excel_data = f.read()

                        st.download_button(
                            label="Download Excel",
                            data=excel_data,
                            file_name="biocine_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        # Remove o arquivo temporário
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Erro ao salvar como Excel: {str(e)}")
        else:
            st.info("Nenhum dado disponível. Importe um arquivo CSV ou gere dados de exemplo.")