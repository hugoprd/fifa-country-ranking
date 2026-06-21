# fifa-country-ranking
A Transformer-based ML model to predict FIFA rankings. It uses a bottom-up approach, weighting player attributes based on league difficulty derived from continental tournament data.

# [PT-BR]

## 🚀 Configuração do Ambiente (Setup)

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

# [US-EN]

## 🚀 Environment Setup

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
