# BioCine - Software de Modelagem Cinética

![BioCine Logo](https://via.placeholder.com/800x150?text=BioCine+-+Modelagem+Cin%C3%A9tica)

## Descrição

BioCine é um software especializado para modelagem cinética e previsão de eficiência no tratamento terciário do soro de leite utilizando microalgas e fungos filamentosos. Esta ferramenta permite:

- Importar e gerenciar dados experimentais
- Aplicar modelos cinéticos de Monod e Logístico
- Analisar a eficiência de remoção de poluentes
- Prever o comportamento do sistema utilizando machine learning
- Otimizar o processo de tratamento
- Visualizar resultados através de gráficos interativos
- Exportar relatórios e análises

## Requisitos do Sistema

- Python 3.8 ou superior
- Bibliotecas Python listadas em `requirements.txt`
- Mínimo 4GB de RAM
- Windows 10/11, macOS ou Linux

## Instalação

### 1. Clone o repositório ou baixe os arquivos

```bash
git clone https://github.com/seunome/biocine.git
cd biocine
```

### 2. Configure um ambiente virtual (recomendado)

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

## Utilização

### Executando a aplicação

```bash
streamlit run app.py
```

Após executar o comando acima, o software será iniciado e abrirá automaticamente no seu navegador padrão. Caso isso não aconteça, acesse:

```
http://localhost:8501
```

### Fluxo de Trabalho

1. **Entrada de Dados**:
   - Utilize os dados de exemplo para demonstração
   - Carregue seus próprios dados em formato CSV
   - Insira dados manualmente

2. **Modelagem Cinética**:
   - Ajuste modelos de Monod e Logístico aos seus dados
   - Visualize o ajuste dos modelos
   - Obtenha parâmetros cinéticos

3. **Machine Learning**:
   - Treine modelos para prever parâmetros
   - Otimize o processo de tratamento
   - Analise a importância dos fatores

4. **Visualização e Resultados**:
   - Visualize dados consolidados
   - Compare diferentes modelos
   - Exporte relatórios com os resultados

## Estrutura do Projeto

```
biocine/
│
├── app.py                # Aplicação principal Streamlit
├── models.py             # Implementação dos modelos cinéticos e ML
├── visualization.py      # Funções de visualização
├── data_utils.py         # Processamento de dados
├── requirements.txt      # Dependências do projeto
└── README.md             # Este arquivo
```

## Fundamentos Teóricos

### Modelo de Monod

O modelo de Monod descreve a relação entre a taxa específica de crescimento dos micro-organismos e a concentração de substrato limitante:

```
μ = μmax * S / (Ks + S)
```

Onde:
- μ: Taxa específica de crescimento
- μmax: Taxa máxima específica de crescimento
- S: Concentração de substrato
- Ks: Constante de meia saturação

### Modelo Logístico

O modelo Logístico descreve o crescimento de populações biológicas em ambientes com recursos limitados:

```
dX/dt = μmax * X * (1 - X/Xmax)
```

Ou em sua forma integrada:

```
X(t) = Xmax / (1 + ((Xmax - X0) / X0) * exp(-μmax * t))
```

Onde:
- X: Concentração de biomassa
- X0: Concentração inicial de biomassa
- Xmax: Capacidade máxima do sistema
- μmax: Taxa máxima de crescimento
- t: Tempo

## Referências

- HE, Y. et al. (2016). Analysis and model delineation of marine microalgae growth and lipid accumulation in flat-plate photobioreactor.
- SOARES, A.P.M.R. et al. (2020). Random Forest as a promising application to predict basic-dye biosorption process using orange waste.
- MALTSEV, Y. & MALTSEVA, K. (2021). Fatty acids of microalgae: Diversity and applications.

## Suporte

Para relatar problemas ou solicitar novos recursos, abra uma issue no GitHub ou entre em contato pelo email:

suporte@biocine.com.br

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.