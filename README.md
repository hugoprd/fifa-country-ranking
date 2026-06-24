# fifa-country-ranking

> Modelo de Machine Learning baseado em Transformers para estimar a força de seleções nacionais de futebol a partir dos atributos individuais e da sinergia entre os jogadores em seus clubes, em vez do histórico de resultados utilizado pela FIFA.

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

### 1. Visão Geral

Este projeto estima a posição de uma seleção nacional de futebol em um ranking mundial a partir de uma fonte de informação diferente da utilizada pela FIFA. Em vez de basear a avaliação no histórico de resultados da própria seleção, o modelo parte do desempenho individual de cada jogador em seu clube e da sinergia (entrosamento) entre jogadores do mesmo país que atuam juntos fora das datas de seleção.

### 2. Como Funciona o Ranking Oficial da FIFA

Desde agosto de 2018, o ranking oficial da FIFA é calculado por um sistema baseado no método Elo, originalmente desenvolvido para o xadrez. A cada partida oficial, a pontuação de uma seleção é atualizada pela fórmula:

`P = P_anterior + I × (W − We)`

Onde:
- `I` é o peso de importância da partida (de 5 pontos em amistosos fora das datas FIFA a 60 pontos nas fases decisivas da Copa do Mundo);
- `W` é o resultado da partida (1 para vitória, 0,5 para empate, 0 para derrota; pênaltis contam como vitória ou derrota);
- `We` é o resultado esperado, calculado exclusivamente a partir da diferença de pontuação Elo entre as duas seleções.

Esse sistema não utiliza nenhum atributo individual de jogador, nem informações de desempenho em clubes: ele depende somente do histórico de resultados oficiais da própria seleção. Por ser uma fórmula matemática determinística e bem definida, um modelo de Machine Learning dificilmente agregaria valor caso utilizasse os mesmos insumos da FIFA. Por esse motivo, este projeto adota uma fonte de dados completamente diferente, descrita a seguir.

### 3. Abordagem do Projeto: Sinergia entre Jogadores

A hipótese investigada é a seguinte: é possível estimar a força real de uma seleção observando exclusivamente a qualidade individual de seus jogadores e a sinergia construída entre eles em seus clubes, sem usar o histórico de resultados da seleção? A motivação é que uma seleção formada por jogadores individualmente fortes que nunca jogaram juntos pode ter desempenho inferior ao de outra com jogadores menos célebres, porém mais entrosados entre si.

Para medir a competência de um jogador, isoladamente e em conjunto com outro, foi adotado o saldo de gols da partida como métrica, em vez da participação direta em gols (marcação ou assistência). Essa escolha evita um viés sistemático contra jogadores de posições defensivas, que raramente participam diretamente de gols, mas cuja qualidade impacta diretamente o resultado da partida.

Cada jogador também recebe um peso proporcional à dificuldade da confederação em que seu clube atua (UEFA, CONMEBOL, CAF, CONCACAF, AFC), calculado a partir do desempenho histórico de cada confederação na Copa do Mundo de Clubes. Isso evita equiparar um jogador irrelevante em um clube da UEFA a um jogador irrelevante em um clube de uma confederação historicamente mais fraca.

### 4. O Modelo de Machine Learning

Trata-se de um modelo de **aprendizado supervisionado de regressão** — não é um modelo de classificação nem de clusterização. A cada seleção é associado um único valor numérico contínuo, que representa sua força estimada, treinado por retropropagação do erro minimizando o erro quadrático médio (MSE) contra um valor de referência calculado a partir da sinergia agregada entre jogadores compatriotas.

A arquitetura é um **Transformer Encoder** — o mesmo bloco básico de auto-atenção usado em LLMs, porém aplicado a um problema de regressão sobre dados estruturados, e não à geração de texto. Cada um dos 11 jogadores principais de uma seleção é tratado como um "token", representado por um vetor de atributos (partidas jogadas, vitórias ponderadas, saldo de gols ponderado, percentual de vitórias). Em vez de uma atenção genérica entre os jogadores, a auto-atenção é tendenciada por uma matriz de sinergia par a par, informando ao modelo quais jogadores já construíram entrosamento em seus clubes. Seleções com menos de 11 jogadores mapeados recebem posições de preenchimento, explicitamente excluídas do cálculo via máscara dedicada.

Resumo técnico:
- **Tipo de aprendizado:** supervisionado;
- **Tipo de tarefa:** regressão (saída numérica contínua);
- **Arquitetura:** Transformer Encoder com atenção tendenciada por grafo de sinergia;
- **Função de perda:** erro quadrático médio (MSE);
- **Otimizador:** Adam, com agendamento de taxa de aprendizado por cosseno e recorte de gradiente;
- **Seleção de hiperparâmetros:** validação cruzada em 5 partições sobre combinações de dimensão de embedding, número de cabeças de atenção e número de camadas.

A explicação detalhada da arquitetura está no relatório técnico do projeto.

### 5. Pipeline de Dados

Os dados vêm do FootyStats (competições globais), da Wikipedia (competições continentais) e do dataset "Football Data from Transfermarkt" no Kaggle (estatísticas de jogadores e partidas). O pipeline segue três camadas:

- **Raw:** dados originais, sem alterações.
- **Processed:** cálculo dos pesos de confederação a partir da Copa do Mundo de Clubes e mapeamento de cada clube/jogador à sua confederação.
- **Refined:** geração dos três datasets finais usados no treinamento — `ml_individual_features.csv` (eficiência individual), `ml_national_synergy_features.csv` (matriz de sinergia entre pares de compatriotas) e `ml_national_team_ranking.csv` (alvo de treinamento).

> **Limitações encontradas:** dados confiáveis e granulares só estavam disponíveis entre 2018 e 2025; e foi matematicamente difícil estabelecer pesos perfeitos entre confederações de continentes diferentes, dada a escassez de confrontos diretos fora da Copa do Mundo de Clubes.

### 6. Resultados: Modelo vs Ranking Oficial da FIFA

Após o treinamento, o modelo gerou seu próprio ranking global de seleções, comparado a seguir ao Top 10 oficial da FIFA:

| ML Rank | Seleção | ML Synergy Power | FIFA Rank | ML vs FIFA (Delta) |
|---------|-------------|------------------|-----------|---------------------|
| 1 | Alemanha | 1.114792 | 8.0 | ↑ +7 |
| 2 | Brasil | 0.898887 | 5.0 | ↑ +3 |
| 3 | Espanha | 0.801684 | 3.0 | — = |
| 4 | Inglaterra | 0.770999 | 4.0 | — = |
| 5 | França | 0.723544 | 2.0 | ↓ -3 |
| 6 | Países Baixos | 0.681522 | 7.0 | ↑ +1 |
| 7 | Portugal | 0.658164 | 9.0 | ↑ +2 |
| 8 | Bélgica | 0.640771 | 10.0 | ↑ +2 |
| 9 | Argentina | 0.618244 | 1.0 | ↓ -8 |
| 10 | Escócia | 0.548400 | Fora do Top 10 | N/A |

**Legenda:** ↑ o modelo valoriza mais o país (pela sinergia) do que a FIFA | ↓ o modelo valoriza menos | — = mesma posição.

### 7. Trabalhos Futuros

Uma direção futura para o projeto é substituir, ou complementar, o saldo de gols da equipe como métrica de desempenho individual por notas de desempenho atribuídas a cada jogador em cada partida específica, disponíveis em bases de dados como o SofaScore ou outras fontes gratuitas e de fácil acesso que ofereçam esse tipo de informação. Por serem calculadas partida a partida e associadas diretamente à atuação individual do jogador, essas notas tendem a capturar com mais precisão aspectos do desempenho não refletidos no saldo de gols da equipe, tornando a medição de eficiência individual e de sinergia mais fiel ao desempenho real de cada jogador.

### 8. Como Executar o Projeto

**Pré-requisitos:** Git, acesso à internet (para scraping e download de dados) e uma conta no Kaggle (necessária para a etapa 8.3).

#### 8.1. Instalação do `uv`

```bash
# Linux ou macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 8.2. Clonagem do repositório

```bash
git clone https://github.com/hugoprd/fifa-country-ranking.git
cd fifa-country-ranking
```

#### 8.3. Preparação manual de dados (obrigatória antes do pipeline)

Dois insumos não são obtidos automaticamente pelos scripts e precisam existir antes de rodar o pipeline:

1. **Dataset Transfermarkt (Kaggle):** baixar `appearances.csv`, `players.csv`, `clubs.csv`, `competitions.csv` e `games.csv` em <https://www.kaggle.com/datasets/davidcariboo/player-scores> e copiá-los para `data/raw/`.
2. **`cwc_confederations_map.csv`:** arquivo com as colunas `team_name` e `confederation`, posicionado em `data/external_metadata/`. Nenhum script do repositório gera esse arquivo automaticamente; ele precisa ser criado ou fornecido manualmente antes de executar `process_data.py`.

#### 8.4. Execução automática completa

```bash
./cmd/environment_setup.sh
./cmd/run_all.sh
```

> **Aviso para usuários Windows:** os scripts `.bat` não foram exaustivamente testados; em caso de falha, prefira a execução manual descrita abaixo.

```bat
cmd\environment_setup.bat
cmd\run_all.bat
```

#### 8.5. Execução manual por etapas (alternativa)

```bash
# 1. Pipeline de dados (na ordem)
python data/extract_external_metadata.py
python data/extract_data.py
python data/process_data.py
python data/refine_data.py

# 2. Pipeline de modelo (na ordem)
python ml_model_scripts/architecture_loader.py
python ml_model_scripts/train_model.py
python ml_model_scripts/generate_fifa_ranking.py
```

#### 8.6. Saída esperada

Ao final, o ranking gerado pelo modelo fica disponível em `ml_fifa_ranking.csv`, na raiz do projeto, e a comparação com o Top 10 oficial da FIFA é impressa no terminal.

---

## EN-US

### 1. Overview

This project estimates a national football team's position in a global ranking using a source of information different from the one used by FIFA. Instead of basing the evaluation on the national team's own match history, the model relies on the individual performance of each player at their club and the synergy (chemistry) between players from the same country who play together outside national-team duty.

### 2. How the Official FIFA Ranking Works

Since August 2018, the official FIFA ranking has been calculated using a system based on the Elo method, originally developed for chess. After every official match, a team's score is updated using the formula:

`P = P_before + I × (W − We)`

Where:
- `I` is the match importance weight (from 5 points for friendlies outside FIFA dates to 60 points for decisive World Cup stages);
- `W` is the match result (1 for a win, 0.5 for a draw, 0 for a loss; penalty shoot-outs count as a win or loss);
- `We` is the expected result, calculated solely from the Elo point difference between the two teams.

This system uses no individual player attributes and no club-level performance data: it depends only on the national team's own official match history. Because it is a deterministic, well-defined mathematical formula, a Machine Learning model would add little value if fed the same inputs as FIFA. For this reason, this project relies on a completely different data source, described below.

### 3. Project Approach: Player Synergy

The hypothesis investigated here is the following: can the real strength of a national team be estimated by looking exclusively at the individual quality of its players and the synergy built between them at club level, without using the national team's match history? The motivation is that a team made up of individually strong players who never played together can underperform compared to a less famous, but better-synchronized, team.

To measure a player's competence, both individually and jointly with another player, the goal difference of the match was used as the metric, instead of direct goal involvement (scoring or assisting). This choice avoids a systematic bias against defensive players, who rarely participate directly in goals but whose quality directly affects the match outcome.

Each player also receives a weight proportional to the difficulty of the confederation their club competes in (UEFA, CONMEBOL, CAF, CONCACAF, AFC), calculated from each confederation's historical performance in the Club World Cup. This avoids treating an irrelevant player at a UEFA club the same as an irrelevant player at a club from a historically weaker confederation.

### 4. The Machine Learning Model

This is a **supervised regression model** — not a classification model and not a clustering model. Each national team is mapped to a single continuous numeric value representing its estimated strength, trained via backpropagation by minimizing the mean squared error (MSE) against a reference value computed from the aggregated synergy between compatriot players.

The architecture is a **Transformer Encoder** — the same self-attention building block used in LLMs, but applied to a regression problem over structured data rather than text generation. Each of a team's 11 main players is treated as a "token", represented by a feature vector (matches played, weighted wins, weighted goal difference, win-rate percentage). Instead of generic attention between players, self-attention is biased by a pairwise synergy matrix that tells the model which players have already built chemistry at their clubs. Teams with fewer than 11 mapped players receive padding slots, explicitly excluded from the computation through a dedicated mask.

Technical summary:
- **Learning type:** supervised;
- **Task type:** regression (continuous numeric output);
- **Architecture:** Transformer Encoder with attention biased by a synergy graph;
- **Loss function:** mean squared error (MSE);
- **Optimizer:** Adam, with cosine learning-rate scheduling and gradient clipping;
- **Hyperparameter selection:** 5-fold cross-validation over combinations of embedding dimension, number of attention heads, and number of layers.

A detailed explanation of the architecture is provided in the project's technical report.

### 5. Data Pipeline

Data comes from FootyStats (global competitions), Wikipedia (continental competitions), and the "Football Data from Transfermarkt" Kaggle dataset (player and match statistics). The pipeline follows three layers:

- **Raw:** original data, unmodified.
- **Processed:** confederation weight calculation from the Club World Cup and mapping of each club/player to its confederation.
- **Refined:** generation of the three final training datasets — `ml_individual_features.csv` (individual efficiency), `ml_national_synergy_features.csv` (pairwise synergy matrix between compatriots) and `ml_national_team_ranking.csv` (training target).

> **Known limitations:** reliable, granular data was only available between 2018 and 2025; and it was mathematically difficult to establish perfect weights between confederations from different continents, given the scarcity of direct matches outside the Club World Cup.

### 6. Results: Model vs Official FIFA Ranking

After training, the model produced its own global national-team ranking, compared below to the official FIFA Top 10:

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

**Legend:** ↑ the model values the country more (due to synergy) than FIFA | ↓ the model values it less | — = same position.

### 7. Future Work

One future direction for the project is to replace, or complement, the team's goal difference as an individual performance metric with per-match player ratings, available from databases such as SofaScore or other free, easily accessible sources that provide this kind of information. Since these ratings are calculated match by match and tied directly to the player's individual performance, they tend to capture aspects of performance not reflected in the team's goal difference more precisely, making the measurement of individual efficiency and synergy more faithful to each player's actual performance.

### 8. How to Run the Project

**Prerequisites:** Git, internet access (for scraping and data download), and a Kaggle account (required for step 8.3).

#### 8.1. Installing `uv`

```bash
# Linux or macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 8.2. Cloning the repository

```bash
git clone https://github.com/hugoprd/fifa-country-ranking.git
cd fifa-country-ranking
```

#### 8.3. Manual data preparation (required before running the pipeline)

Two inputs are not fetched automatically by the scripts and must exist before running the pipeline:

1. **Transfermarkt dataset (Kaggle):** download `appearances.csv`, `players.csv`, `clubs.csv`, `competitions.csv` and `games.csv` from <https://www.kaggle.com/datasets/davidcariboo/player-scores> and copy them into `data/raw/`.
2. **`cwc_confederations_map.csv`:** a file with the columns `team_name` and `confederation`, placed in `data/external_metadata/`. No script in the repository generates this file automatically; it must be created or provided manually before running `process_data.py`.

#### 8.4. Full automatic execution

```bash
./cmd/environment_setup.sh
./cmd/run_all.sh
```

> **Notice for Windows users:** the `.bat` scripts have not been exhaustively tested; if they fail, prefer the manual execution described below.

```bat
cmd\environment_setup.bat
cmd\run_all.bat
```

#### 8.5. Manual step-by-step execution (alternative)

```bash
# 1. Data pipeline (in order)
python data/extract_external_metadata.py
python data/extract_data.py
python data/process_data.py
python data/refine_data.py

# 2. Model pipeline (in order)
python ml_model_scripts/architecture_loader.py
python ml_model_scripts/train_model.py
python ml_model_scripts/generate_fifa_ranking.py
```

#### 8.6. Expected output

At the end, the model's generated ranking is available at `ml_fifa_ranking.csv`, in the project root, and the comparison with the official FIFA Top 10 is printed to the terminal.