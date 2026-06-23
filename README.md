# fifa-country-ranking

> A Transformer-based ML model to predict FIFA rankings. It uses a bottom-up approach, weighting player attributes based on league difficulty derived from continental tournament data.


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

---

## PT-BR

### O Problema: Medindo a Sinergia

O Ranking oficial da FIFA avalia as seleções nacionais com base em um cálculo simples de vitórias, empates e derrotas. No entanto, o futebol moderno é complexo. Uma seleção com 11 superestrelas que nunca jogaram juntas muitas vezes perde para uma equipe menos badalada, mas altamente entrosada.

O objetivo deste projeto é responder a uma pergunta fundamental:

> "Podemos prever a força real de uma seleção nacional medindo exclusivamente a qualidade individual de seus jogadores e a sinergia (entrosamento) entre eles nos clubes?"

---

### Os Dados e Limitações

A construção desse pipeline exigiu a coleta de dados de diversas fontes (FootyStats, Wikipedia, Transfermarkt e FBref). O pipeline consolida essas informações em três bases principais refinadas:

- `ml_individual_features.csv`: Estatísticas de eficiência de cada jogador.
- `ml_national_synergy_features.csv`: O mapeamento de quem joga (ou já jogou) com quem no mesmo clube.
- `ml_national_team_ranking.csv`: O alvo (target) para o treinamento do modelo.

> ⚠️ **Limitações Encontradas:**
>
> Durante a engenharia de dados, esbarramos em algumas restrições do mundo real:
>
> - **Janela de Tempo Escassa:** Foi possível a coleta de dados confiáveis e granulares apenas do período entre 2018 e 2025.
> - **Pesos das Confederações:** Foi matematicamente difícil estabelecer um "peso" perfeito para comparar a dificuldade entre ligas da UEFA (Europa) e CONMEBOL (América do Sul) com as demais confederações, devido à falta de confrontos diretos frequentes entre clubes de continentes diferentes fora do Mundial de Clubes.

---

### Inteligência Artificial: Por que Transformers?

Para resolver o problema da sinergia, foi escolhido a arquitetura **Transformer**, mas aplicada a esportes.

Em vez de processar "palavras em uma frase", nosso modelo processa **"jogadores em um elenco"**.

Foi transformado os dados de sinergia dos jogadores em **Grafos (Matrizes de Adjacência)**. Quando o Transformer usa seu mecanismo de **Self-Attention (Atenção)**, ele olha para essa matriz e entende automaticamente quais jogadores possuem "conexões" prévias em clubes. Assim, o modelo aprende a dar mais peso para pequenos grupos entrosados (ex: um trio de meio-campo que joga no mesmo time há 3 anos).

---

### Resultados: Modelo vs FIFA Oficial

Após o treinamento e otimização de hiperparâmetros, o modelo foi colocado para gerar o seu próprio Ranking Global de Seleções e, depois, foi feita uma comparação com o Top 10 oficial da FIFA.

O modelo destacou seleções **"subestimadas"** pela FIFA (que possuem alta sinergia coletiva) e rebaixou seleções **"superestimadas"** (cheias de estrelas isoladas).

| ML Rank | Country | ML Synergy Power | FIFA Rank | ML vs FIFA (Delta) |
|---------|-------------|------------------|-----------|---------------------|
| 1 | Germany | 1.114792 | 8.0 | ↑ +7 |
| 2 | Brazil | 0.898887 | 5.0 | ↑ +3 |
| 3 | Spain | 0.801684 | 3.0 | — = |
| 4 | England | 0.770999 | 4.0 | — = |
| 5 | France | 0.723544 | 2.0 | ↓ -3 |
| 6 | Netherlands | 0.681522 | 7.0 | ↑ +1 |
| 7 | Portugal | 0.658164 | 9.0 | ↑ +2 |
| 8 | Belgium | 0.640771 | 10.0 | ↑ +2 |
| 9 | Argentina | 0.618244 | 1.0 | ↓ -8 |
| 10 | Scotland | 0.548400 | Outside Top 10 | N/A |

**Legenda:** ↑ O ML valoriza mais o país (pela sinergia) do que a FIFA | ↓ O ML valoriza menos.

---

### Configuração do Ambiente (Setup)

Este projeto utiliza o `uv` como gerenciador de pacotes e ambientes virtuais, garantindo uma instalação rápida e isolada.

#### 1. Instalação do `uv`

Se o `uv` ainda não está instalado, abra o seu terminal:

**No Linux ou macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**No Windows via PowerShell:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Clonagem do repositório

```bash
git clone https://github.com/hugoprd/fifa-country-ranking.git
cd fifa-country-ranking
```

#### 3. Setup Automático

O projeto conta com scripts automatizados para instalar as dependências e rodar todo o pipeline (Dados -> Modelo -> Inferência) com apenas dois comandos.

**No Linux / macOS:**
```bash
./cmd/environment_setup.sh
./cmd/run_all.sh
```

> ⚠️ **Aviso para usuários Windows:** A execução dos scripts batch (`.bat`) automatizados no Windows não foi exaustivamente testada e comportamentos inesperados (erros de caminhos ou de ambiente) podem ocorrer. Caso os scripts falhem, recomenda-se a instalação manual das dependências e a execução individual dos módulos Python.

```bat
cmd/environment_setup.bat
cmd/run_all.bat
```

---

## EN-US

### The Problem: Measuring Synergy

The official FIFA Ranking evaluates national teams based on a simple calculation of wins, draws, and losses. However, modern football is complex. A national team with 11 superstars who have never played together often loses to a less famous but highly synergized team.

The goal of this project is to answer a fundamental question:

> "Can we predict the true strength of a national team by measuring exclusively the individual quality of its players and the synergy (chemistry) between them from their club experience?"

---

### Data and Limitations

Building this pipeline required gathering data from multiple sources (FootyStats, Wikipedia, Transfermarkt and FBref). The pipeline consolidates this information into three main refined datasets:

- `ml_individual_features.csv`: Individual player efficiency statistics.
- `ml_national_synergy_features.csv`: The mapping of who plays (or has played) with whom in the same club.
- `ml_national_team_ranking.csv`: The target variable for the model training.

> ⚠️ **Known Limitations:**
>
> During the data engineering phase, we encountered some real-world constraints:
>
> - **Scarce Time Window:** It was only able to gather reliable and granular data from the period between 2018 and 2025.
> - **Confederation Weights:** It was mathematically challenging to establish a perfect "weight" to compare the difficulty between UEFA (Europe) and CONMEBOL (South America) leagues against other confederations, due to the lack of frequent direct matches between clubs from different continents outside the Club World Cup.

---

### Artificial Intelligence: Why Transformers?

To solve the synergy problem, the **Transformer** architecture was chosen, but applied to sports.

Instead of processing "words in a sentence", our model processes **"players in a squad"**.

It was transformed the players' synergy data into **Graphs (Adjacency Matrices)**. When the Transformer uses its **Self-Attention** mechanism, it looks at this matrix and automatically understands which players have previous "connections" in clubs. Thus, the model learns to give more weight to small, highly synergized groups (e.g., a midfield trio that has played on the same team for 3 years).

---

### Results: Model vs Official FIFA

After training and hyperparameter optimization, it was deployed our model to generate its own Global National Team Ranking and then a comparison was made with the official FIFA Top 10.

The model highlighted **"underrated"** teams by FIFA (which possess high collective synergy) and downgraded **"overrated"** teams (packed with isolated stars).

| ML Rank | Country | ML Synergy Power | FIFA Rank | ML vs FIFA (Delta) |
|---------|-------------|------------------|-----------|---------------------|
| 1 | Germany | 1.114792 | 8.0 | ↑ +7 |
| 2 | Brazil | 0.898887 | 5.0 | ↑ +3 |
| 3 | Spain | 0.801684 | 3.0 | — = |
| 4 | England | 0.770999 | 4.0 | — = |
| 5 | France | 0.723544 | 2.0 | ↓ -3 |
| 6 | Netherlands | 0.681522 | 7.0 | ↑ +1 |
| 7 | Portugal | 0.658164 | 9.0 | ↑ +2 |
| 8 | Belgium | 0.640771 | 10.0 | ↑ +2 |
| 9 | Argentina | 0.618244 | 1.0 | ↓ -8 |
| 10 | Scotland | 0.548400 | Outside Top 10 | N/A |

**Legend:** ↑ ML values the country more (due to synergy) than FIFA | ↓ ML values it less.

---

### Environment Setup

This project uses `uv` as its package and virtual environment manager, ensuring an extremely fast and isolated installation.

#### 1. `uv` Installation

If you don't have `uv` installed yet, open your terminal:

**On Linux or macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows via PowerShell:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Repository Clone

```bash
git clone https://github.com/hugoprd/fifa-country-ranking.git
cd fifa-country-ranking
```

#### 3. Automatic Setup

The project features automated scripts to install dependencies and run the entire pipeline (Data -> Model -> Inference) with just two commands.

**On Linux / macOS:**
```bash
./cmd/environment_setup.sh
./cmd/run_all.sh
```

> ⚠️ **Notice for Windows users:** The execution of the automated batch (`.bat`) scripts on Windows has not been exhaustively tested and unexpected behavior (path or environment errors) may occur. If the scripts fail, it is recommended to manually install the dependencies and execute the Python modules individually.

```bat
cmd/environment_setup.bat
cmd/run_all.bat
```