# BioCine - Software de Modelagem Cinética de Bioprocessos

## Descrição
BioCine é uma ferramenta de software desenvolvida para a modelagem cinética do processo de tratamento terciário em batelada/semicontínuo do soro do leite por microalgas e fungos filamentosos. O software implementa modelos cinéticos clássicos (Monod e Logístico) e técnicas de aprendizado de máquina para prever e otimizar a eficiência do processo.

## Características Principais
- Modelagem cinética usando modelo de Monod
- Modelagem cinética usando modelo Logístico
- Visualização de dados experimentais
- Importação e processamento de dados em formato CSV
- Cálculo de eficiência de remoção de poluentes (Nitrogênio, Fósforo, DQO)
- Visualização de crescimento de biomassa
- Predição usando Random Forest
- Avaliação de modelos (MSE, R²)
- Visualização da importância de características
- Exportação de relatório

## Instalação

```bash
# Clone o repositório
git clone https://github.com/username/biocine.git
cd biocine

# Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

## Uso
Para iniciar o aplicativo:

```bash
streamlit run app.py
```

## Estrutura do Projeto
```
biocine/
├── app.py                    # Ponto de entrada principal (Streamlit)
├── requirements.txt          # Dependências do projeto
├── README.md                 # Documentação do projeto
├── config/                   # Configurações do aplicativo
├── data/                     # Diretório para dados
├── models/                   # Implementações de modelos
├── tests/                    # Testes automatizados
├── ui/                       # Interface de usuário modularizada
└── utils/                    # Utilitários
```

## Contribuição
Este projeto é parte de uma pesquisa acadêmica. Para contribuir, por favor entre em contato com os autores.

## Licença
Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Citação
Se você usar este software em sua pesquisa, por favor cite:

```
Nascimento, M. A. A. (2024). Modelagem Cinética do Processo de Tratamento Terciário em Batelada/Semicontínuo do Soro do Leite por Microalgas e Fungos Filamentosos. Relatório Parcial de Iniciação à Pesquisa, Ciclo 2024-2025.
```