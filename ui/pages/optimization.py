import pandas as pd
import streamlit as st
from models.advanced_optimization import ProcessOptimizer


def show_advanced_optimization():
    """Página de Otimização Avançada de Processos"""
    st.markdown("<h2 class='sub-header'>Otimização Avançada de Bioprocessos</h2>", unsafe_allow_html=True)

    # Verifica se há dados disponíveis
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Nenhum dado disponível. Por favor, vá para a seção 'Entrada de Dados' primeiro.")
        return

    # Inicializa o otimizador se não existir
    if 'process_optimizer' not in st.session_state:
        st.session_state.process_optimizer = ProcessOptimizer()

    optimizer = st.session_state.process_optimizer

    # Carrega modelos treinados da sessão
    if 'ml_models' in st.session_state and st.session_state.ml_models:
        available_models = list(st.session_state.ml_models.keys())
        st.info(f"Encontrados {len(available_models)} modelos: {', '.join(available_models)}")

        # Se não houver modelos no otimizador, carrega os modelos da sessão
        if not optimizer.models:
            for target, model in st.session_state.ml_models.items():
                optimizer.add_model(target, model)
                st.success(f"Modelo para '{target}' carregado no otimizador")

    # Verifica se há modelos disponíveis
    if not optimizer.models:
        st.warning("Nenhum modelo disponível. Treine modelos primeiro na seção 'Machine Learning'.")
        # Fornece dicas para o usuário
        st.info("""
        Para usar esta funcionalidade:
        1. Vá para a seção 'Machine Learning'
        2. Selecione um alvo para prever (ex: biomassa, eficiencia_remocao)
        3. Selecione características para o modelo
        4. Treine o modelo
        5. Retorne a esta página
        """)
        return

    # Verifica se há parâmetros definidos
    if not optimizer.parameter_bounds:
        st.subheader("Configuração de Parâmetros")

        # Tenta extrair nomes de parâmetros do primeiro modelo
        try:
            first_model = next(iter(optimizer.models.values()))

            # Tenta diferentes maneiras de obter nomes de características
            if hasattr(first_model, 'feature_names_in_'):
                feature_names = list(first_model.feature_names_in_)
            elif hasattr(first_model, 'feature_names'):
                feature_names = list(first_model.feature_names)
            else:
                # Fallback: extrai das variáveis de sessão
                data = st.session_state.data
                feature_names = [col for col in data.columns if col not in optimizer.models.keys()]

            # Apresenta interface para definir limites
            st.write("Defina os limites para os parâmetros de otimização:")

            param_bounds = {}
            for feature in feature_names:
                col1, col2 = st.columns(2)

                with col1:
                    lower = st.number_input(f"Mínimo para {feature}", value=0.0, key=f"min_{feature}")

                with col2:
                    # Define valor máximo padrão baseado nos dados quando disponíveis
                    if 'data' in st.session_state and feature in st.session_state.data.columns:
                        max_val = float(st.session_state.data[feature].max() * 1.5)
                        # Limita a um valor razoável
                        max_val = min(max_val, 1000)
                    else:
                        max_val = 100.0

                    upper = st.number_input(f"Máximo para {feature}", value=max_val, key=f"max_{feature}")

                param_bounds[feature] = (lower, upper)

            # Botão para definir parâmetros
            if st.button("Definir Parâmetros"):
                optimizer.set_parameter_bounds(param_bounds)
                st.success("Parâmetros definidos com sucesso!")
                st.rerun()  # Recarrega a página

            return

        except Exception as e:
            st.error(f"Erro ao configurar parâmetros: {str(e)}")

            # Interface manual para definição de parâmetros
            st.subheader("Configuração Manual de Parâmetros")

            with st.form("manual_params"):
                n_params = st.number_input("Número de parâmetros", 1, 10, 3)

                params = {}
                for i in range(int(n_params)):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        param_name = st.text_input(f"Parâmetro {i + 1}", f"param_{i + 1}")

                    with col2:
                        lower = st.number_input(f"Mínimo {i + 1}", value=0.0)

                    with col3:
                        upper = st.number_input(f"Máximo {i + 1}", value=100.0)

                    params[param_name] = (lower, upper)

                submit_button = st.form_submit_button("Definir Parâmetros")

                if submit_button:
                    optimizer.set_parameter_bounds(params)
                    st.success("Parâmetros definidos com sucesso!")
                    st.rerun()  # Recarrega a página

            return

    # Interface principal de otimização
    # Tabs para diferentes funções
    tab1, tab2, tab3, tab4 = st.tabs([
        "Otimização Simples",
        "Otimização Multiobjetivo",
        "Superfície de Resposta",
        "Planejamento Experimental"
    ])

    with tab1:
        st.subheader("Otimização Simples")

        # Seleção de alvo
        target = st.selectbox(
            "Selecione o alvo para otimização",
            list(optimizer.models.keys())
        )

        # Modo de otimização
        mode = st.radio(
            "Modo de otimização",
            ["Maximizar", "Minimizar"],
            horizontal=True
        )

        # Método de otimização
        method = st.selectbox(
            "Método de otimização",
            ["SLSQP", "genetic", "annealing"],
            index=0,
            help="SLSQP: gradiente, genetic: algoritmo genético, annealing: recozimento simulado"
        )

        # Parâmetros do algoritmo
        if method == "genetic" or method == "annealing":
            max_iter = st.slider("Número máximo de iterações", 10, 1000, 100)
            population = st.slider("Tamanho da população", 10, 200, 50)
        else:
            max_iter = st.slider("Número máximo de iterações", 10, 1000, 100)
            population = 50

        # Botão para executar otimização
        if st.button("Executar Otimização", key="opt_simple"):
            with st.spinner("Otimizando..."):
                try:
                    # Configura a função objetivo
                    optimizer.set_objective(
                        mode=mode.lower(),
                        target=target
                    )

                    # Executa a otimização
                    result = optimizer.optimize_process(
                        method=method,
                        max_iter=max_iter,
                        population=population
                    )

                    # Exibe resultados
                    st.success(f"Otimização concluída: {result['message']}")

                    # Parâmetros ótimos
                    st.subheader("Parâmetros Ótimos")
                    params_df = pd.DataFrame(
                        [result['optimal_parameters']]
                    )
                    st.dataframe(params_df)

                    # Valor ótimo
                    st.metric(
                        f"Valor Ótimo ({target})",
                        f"{result['optimal_value']:.4f}"
                    )

                except Exception as e:
                    st.error(f"Erro na otimização: {str(e)}")

    with tab2:
        st.subheader("Otimização Multiobjetivo")

        # Seleção de alvos
        available_targets = list(optimizer.models.keys())
        selected_targets = st.multiselect(
            "Selecione os alvos para otimização multiobjetivo",
            available_targets,
            default=available_targets[:min(2, len(available_targets))]
        )

        if len(selected_targets) < 2:
            st.warning("Selecione pelo menos dois alvos para otimização multiobjetivo.")
        else:
            # Pesos para cada alvo
            st.subheader("Pesos dos Objetivos")

            weights = {}
            cols = st.columns(len(selected_targets))

            for i, target in enumerate(selected_targets):
                with cols[i]:
                    weights[target] = st.slider(
                        f"Peso de {target}",
                        0.0, 1.0, 1.0 / len(selected_targets),
                        0.01
                    )

            # Número de pontos Pareto
            n_points = st.slider(
                "Número de pontos Pareto",
                5, 100, 20,
                help="Mais pontos exploram melhor a fronteira Pareto, mas aumentam o tempo de cálculo"
            )

            # Botão para executar otimização
            if st.button("Executar Otimização Multiobjetivo", key="opt_multi"):
                with st.spinner("Calculando fronteira Pareto..."):
                    try:
                        # Executa otimização multiobjetivo
                        pareto_df = optimizer.multi_objective_optimization(
                            target_weights=weights,
                            n_points=n_points
                        )

                        # Exibe resultados
                        st.success(f"Otimização multiobjetivo concluída. Encontrados {len(pareto_df)} pontos Pareto.")

                        # Tabela de resultados
                        st.subheader("Pontos Pareto")

                        # Mostra apenas colunas importantes
                        show_cols = optimizer.parameter_names.copy()
                        show_cols.extend([f"{t}_value" for t in selected_targets])
                        st.dataframe(pareto_df[show_cols])

                        # Gráfico de dispersão para 2 objetivos
                        if len(selected_targets) == 2:
                            import plotly.express as px

                            obj1 = f"{selected_targets[0]}_value"
                            obj2 = f"{selected_targets[1]}_value"

                            fig = px.scatter(
                                pareto_df,
                                x=obj1,
                                y=obj2,
                                title="Fronteira Pareto",
                                labels={
                                    obj1: selected_targets[0],
                                    obj2: selected_targets[1]
                                },
                                hover_data=optimizer.parameter_names
                            )

                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Erro na otimização multiobjetivo: {str(e)}")

    with tab3:
        st.subheader("Superfície de Resposta")

        # Seleção de alvo
        target = st.selectbox(
            "Selecione o alvo para a superfície",
            list(optimizer.models.keys()),
            key="surf_target"
        )

        # Seleção de parâmetros
        st.subheader("Parâmetros para a Superfície")

        col1, col2 = st.columns(2)

        with col1:
            param1 = st.selectbox(
                "Parâmetro 1",
                optimizer.parameter_names,
                index=0
            )

        with col2:
            param2 = st.selectbox(
                "Parâmetro 2",
                optimizer.parameter_names,
                index=min(1, len(optimizer.parameter_names) - 1)
            )

        # Resolução
        resolution = st.slider(
            "Resolução da grade",
            10, 100, 30,
            help="Maior resolução = mais detalhes, mais tempo de cálculo"
        )

        # Valores fixos para outros parâmetros
        if len(optimizer.parameter_names) > 2:
            st.subheader("Valores de outros parâmetros")

            fixed_params = {}
            other_params = [p for p in optimizer.parameter_names if p not in [param1, param2]]

            for param in other_params:
                bounds = optimizer.parameter_bounds[param]
                default_value = (bounds[0] + bounds[1]) / 2

                fixed_params[param] = st.slider(
                    param,
                    float(bounds[0]),
                    float(bounds[1]),
                    float(default_value)
                )
        else:
            fixed_params = None

        # Opções de visualização
        viz_type = st.radio(
            "Tipo de Visualização",
            ["Contorno", "Superfície 3D"],
            horizontal=True
        )

        # Botão para gerar visualização
        if st.button("Gerar Visualização", key="gen_surf"):
            with st.spinner("Calculando superfície de resposta..."):
                try:
                    if viz_type == "Contorno":
                        # Gera gráfico de contorno
                        fig = optimizer.create_contour_plot(
                            param1=param1,
                            param2=param2,
                            resolution=resolution,
                            target=target,
                            fixed_params=fixed_params
                        )
                    else:
                        # Gera superfície 3D
                        fig = optimizer.create_response_surface(
                            param1=param1,
                            param2=param2,
                            resolution=resolution,
                            target=target,
                            fixed_params=fixed_params
                        )

                    # Exibe a figura
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Erro ao gerar visualização: {str(e)}")

    with tab4:
        st.subheader("Planejamento Experimental")

        # Tipo de planejamento
        design_type = st.selectbox(
            "Tipo de Planejamento",
            [
                "central_composite",
                "full_factorial",
                "fractional_factorial",
                "box_behnken",
                "latin_hypercube"
            ],
            index=0,
            help="Tipo de planejamento experimental"
        )

        # Pontos centrais
        center_points = st.slider(
            "Pontos Centrais",
            1, 5, 1,
            help="Número de replicações no ponto central"
        )

        # Botão para gerar planejamento
        if st.button("Gerar Planejamento", key="gen_doe"):
            with st.spinner("Gerando planejamento experimental..."):
                try:
                    # Gera o planejamento
                    doe_df = optimizer.design_of_experiments(
                        design_type=design_type,
                        center_points=center_points
                    )

                    # Exibe o planejamento
                    st.subheader("Pontos Experimentais")
                    st.dataframe(doe_df[optimizer.parameter_names])

                    # Seleção de alvo para avaliar
                    target = st.selectbox(
                        "Selecione um alvo para avaliar o planejamento",
                        list(optimizer.models.keys()),
                        key="doe_eval_target"
                    )

                    # Botão para avaliar
                    if st.button("Avaliar Planejamento", key="eval_doe"):
                        with st.spinner("Avaliando planejamento..."):
                            # Avalia com o modelo selecionado
                            results_df = optimizer.evaluate_doe_predictions(target)

                            # Exibe resultados
                            st.subheader("Resultados Previstos")
                            st.dataframe(results_df[[*optimizer.parameter_names, target]])

                            # Plota análise do planejamento
                            fig_dict = optimizer.plot_doe_results(results_df)

                            # Exibe gráficos
                            if isinstance(fig_dict, dict):
                                if 'main_effects' in fig_dict:
                                    st.subheader("Efeitos Principais")
                                    st.plotly_chart(fig_dict['main_effects'], use_container_width=True)

                                if 'interaction_effects' in fig_dict:
                                    st.subheader("Interações")
                                    st.plotly_chart(fig_dict['interaction_effects'], use_container_width=True)

                                if 'pareto' in fig_dict:
                                    st.subheader("Pareto de Efeitos")
                                    st.plotly_chart(fig_dict['pareto'], use_container_width=True)
                            else:
                                st.plotly_chart(fig_dict, use_container_width=True)

                except Exception as e:
                    st.error(f"Erro ao gerar planejamento: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())