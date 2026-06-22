# fifa-country-ranking
A Transformer-based ML model to predict FIFA rankings. It uses a bottom-up approach, weighting player attributes based on league difficulty derived from continental tournament data.

<div align="center">

![uv](https://img.shields.io/badge/uv-261230?style=for-the-badge&logo=astral&logoColor=D7FF64)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

![BeautifulSoup4](https://img.shields.io/badge/BeautifulSoup4-2b2d42?style=for-the-badge)
![soccerdata](https://img.shields.io/badge/soccerdata-2b2d42?style=for-the-badge)
![loguru](https://img.shields.io/badge/loguru-2b2d42?style=for-the-badge)
![tqdm](https://img.shields.io/badge/tqdm-2b2d42?style=for-the-badge)

</div>

- [PT-BR](#pt-br)
- [EN-US](#en-us)

----------------------------------------------------------------------

# [PT-BR]

# Dados

Os dados utilizados neste projeto provêm de múltiplas fontes, incluindo FootyStats (Mundial de Clubes), Wikipedia (competições continentais) e Kaggle/Transfermarkt (estatísticas de jogadores e partidas). 

O pipeline de dados é estruturado em uma Arquitetura Medalhão (Bronze, Prata e Ouro), focado em processar estatísticas brutas para quantificar a eficiência individual e a sinergia entre compatriotas, sempre ponderando pela dificuldade (peso) da liga em que atuam.

Para ver os detalhes completos sobre a origem dos dados, a engenharia de atributos (Feature Engineering) e como rodar o pipeline automatizado de extração e refinamento, **[clique aqui para ler a documentação completa dos Dados (Pasta `data/`)](./data/README.md)**.

# Configuração do Ambiente (Setup)

Este projeto utiliza o [uv](https://github.com/astral-sh/uv) como gerenciador de pacotes e ambientes virtuais, garantindo uma instalação rápida e isolada.

Siga o passo a passo abaixo para configurar o projeto na sua máquina:

### 1. Instalação do `uv`
Se o `uv` ainda não está instalado, abra o seu terminal e execute o comando correspondente ao seu sistema operacional:

**No Linux ou macOS:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**No Windows via PowerShell:**
```bash
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```
*( Após a instalação pode ser necessário reiniciar o terminal para que o comando `uv` seja reconhecido )*

### 2. Clonagem do repositório
Baixe o código para sua máquina e entre na pasta do projeto.

```bash
git clone [https://github.com/hugoprd/fifa-country-ranking.git](https://github.com/hugoprd/fifa-country-ranking.git)
cd fifa-country-ranking
```

### 3. Instalação das dependências e criação do ambiente
Rode o comando `uv` para a criação e sincronização do ambiente com o repositório, na raíz do projeto.

```bash
uv sync
```

### 4. Ativação do ambiente virtual

**No Linux ou macOS:**
```bash
source .venv/bin/activate
```

**No Windows:**
```bash
.venv\Scripts\activate
```

----------------------------------------------------------------------

# [EN-US]

# Data

The data used in this project comes from multiple sources, including FootyStats (Club World Cup), Wikipedia (continental competitions), and Kaggle/Transfermarkt (player and match statistics).

The data pipeline is structured using a Medallion Architecture (Bronze, Silver, and Gold), focused on processing raw statistics to quantify individual efficiency and synergy among compatriots, always weighted by the difficulty (weight) of the league they play in.

For full details on the origin of the data, the Feature Engineering process, and how to run the automated extraction and refinement pipeline, **[click here to read the complete Data documentation (`data/` folder)](./data/README.md)**.

# Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) as its package and virtual environment manager, ensuring an extremely fast and isolated installation.

Follow the step-by-step guide below to set up the project on your machine:

### 1. `uv` instalation
If you don't have `uv` installed yet, open your terminal and run the command corresponding to your operating system:

**On Linux or macOS:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**On Windows via PowerShell:**
```bash
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

*( After installation, you might need to restart your terminal for the `uv` command to be recognized )*

### 2. Repository clone
Download the code to your machine and navigate into the project folder.

```bash
git clone [https://github.com/hugoprd/fifa-country-ranking.git](https://github.com/hugoprd/fifa-country-ranking.git)
cd fifa-country-ranking
```

### 3. Dependencies install and environment creation
Run the `uv` commandto create and synchronize the environment of the repository, into the project root.

```bash
uv sync
```

### 4. Virtual environment activation

**On Linux or macOS:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```