"""
Página de entrada de dados do BioCine

Esta página permite a importação de dados experimentais ou
a geração de dados de exemplo para a modelagem cinética.
"""

import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import import_csv, generate_example_data, validate_data, preprocess_data, save_processed_data


def show_data_input():
    """
    Renderiza a página de entrada de dados
    """
    st.markdown("<h2 class='sub-header'>Entrada de Dados</h2>", unsafe_allow_html=True)

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

                        st.success("Dados processados com sucesso!")
                        st.info("Vá para a guia 'Visualizar Dados' para explorar os dados.")
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
            # Gera os dados
            example_data = generate_example_data(n_samples, seed)

            # Salva os dados no estado da sessão
            st.session_state.data = example_data

            st.success("Dados de exemplo gerados com sucesso!")
            st.info("Vá para a guia 'Visualizar Dados' para explorar os dados.")

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
                    default=["tempo", "biomassa"] if all(
                        col in data.columns for col in ["tempo", "biomassa"]) else data.select_dtypes(
                        include=[np.number]).columns[:2]
                )

                if len(viz_cols) > 0:
                    # Tipo de gráfico
                    plot_type = st.selectbox(
                        "Tipo de Gráfico",
                        ["Linha", "Dispersão", "Histograma", "Boxplot", "Matriz de Correlação"]
                    )

                    # Cria a figura
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if plot_type == "Linha":
                        for col in viz_cols:
                            ax.plot(data["tempo"] if "tempo" in data.columns else data.index,
                                    data[col], marker='o', label=col)
                        ax.set_xlabel("Tempo" if "tempo" in data.columns else "Índice")
                        ax.set_ylabel("Valor")
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.7)

                    elif plot_type == "Dispersão":
                        if len(viz_cols) >= 2:
                            ax.scatter(data[viz_cols[0]], data[viz_cols[1]])
                            ax.set_xlabel(viz_cols[0])
                            ax.set_ylabel(viz_cols[1])
                            ax.grid(True, linestyle='--', alpha=0.7)
                        else:
                            st.warning("Selecione pelo menos duas colunas para o gráfico de dispersão.")

                    elif plot_type == "Histograma":
                        for col in viz_cols:
                            ax.hist(data[col], bins=10, alpha=0.7, label=col)
                        ax.set_xlabel("Valor")
                        ax.set_ylabel("Frequência")
                        ax.legend()

                    elif plot_type == "Boxplot":
                        boxplot_data = [data[col] for col in viz_cols]
                        ax.boxplot(boxplot_data, labels=viz_cols)
                        ax.set_ylabel("Valor")
                        ax.grid(True, linestyle='--', alpha=0.7, axis='y')

                    elif plot_type == "Matriz de Correlação":
                        corr_matrix = data[viz_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)

                    # Mostra o gráfico
                    st.pyplot(fig)

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