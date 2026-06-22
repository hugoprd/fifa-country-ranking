# [PT-BR]

# 1. FONTES DOS DADOS

A base de dados utilizada para o projeto provem do FootyStats para as competicoes globais, da Wikipedia para competicoes continentais e do dataset "Football Data from Transfermarkt" no Kaggle para estatisticas de jogadores e clubes.

## 1.1. Competicoes Continentais
Foi feito um web scraping em cada pagina da Wikipedia de cada competicao continental para a obtencao dos nomes dos times, suas confederacoes e seus paises.
- CAF: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CAF_countries
- CONMEBOL: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CONMEBOL_countries
- UEFA: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_UEFA_countries
- CONCACAF: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CONCACAF_countries
- AFC: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_AFC_countries

## 1.2. Competicoes Mundiais

### 1.2.1. Mundial de Clubes:
- 2025/2025: https://footystats.org/c-dl.php?type=matches&comp=13878
- 2024/2024: https://footystats.org/c-dl.php?type=matches&comp=13557
- 2023/2023: https://footystats.org/c-dl.php?type=matches&comp=10958
- 2022/2022: https://footystats.org/c-dl.php?type=matches&comp=8830
- 2021/2021: https://footystats.org/c-dl.php?type=matches&comp=7069
- 2020/2020: https://footystats.org/c-dl.php?type=matches&comp=5517
- 2019/2019: https://footystats.org/c-dl.php?type=matches&comp=3370
- 2018/2018: https://footystats.org/c-dl.php?type=matches&comp=1813

### 1.2.2. Copa do Mundo:
- https://fbref.com/en/comps/1/

## 1.3. Jogadores e Partidas
Transfermarkt: https://www.kaggle.com/datasets/davidcariboo/player-scores
*( Desta base, foram utilizadas as tabelas `appearances.csv`, `players.csv`, `clubs.csv`, `competitions.csv` e `games.csv` )*

# 2. USO DOS DADOS E PIPELINE

O objetivo principal desta seção é estruturar e quantificar o desempenho dos jogadores em escala global, alimentando um modelo de Machine Learning capaz de avaliar o real poder das Seleções Nacionais. Para garantir justiça nas métricas, o pipeline de dados foi desenhado em tres camadas (Arquitetura Medalhao):

## 2.1. Camada Raw:
Extração e armazenamento dos dados originais das fontes citadas, sem alterações.

## 2.2. Camada Processed:
1. Cálculo de Pesos das Confederacoes: Utiliza o histórico de confrontos diretos do Mundial de Clubes para calcular a força relativa de cada continente (ex: UEFA, CONMEBOL, CAF), baseando-se na média de pontos por jogo (PPG).
2. Mapeamento Relacional: Cruza as tabelas do Transfermarkt (clubes e competições) para descobrir a confederacao exata de cada clube e definir o peso de dificuldade calculado em cada jogador.

## 2.3. Camada Refined:
Gera os datasets finais para o Machine Learning focados em duas frentes de avaliacao universal (Win Rate e Saldo de Gols), permitindo comparar atacantes e defensores de forma justa:
1. Eficiência Individual: Avalia o impacto isolado de cada jogador, multiplicando suas vitórias e saldo de gols pelo peso de dificuldade da sua liga.
2. Matriz de Sinergia (Grafos): Mapeia duplas de jogadores da mesma nacionalidade que atuam juntos no mesmo clube. Mede o "entrosamento" da parceria no nivel de clubes para prever a quimica que entregarao juntos na Selecao Nacional.

----------------------------------------------------------------------

# [EN-US]

# 1. DATA SOURCES

The database used for this project is sourced from FootyStats for global competitions, Wikipedia for continental competitions, and the "Football Data from Transfermarkt" Kaggle dataset for player and match statistics.

## 1.1. Continental Competitions
Web scraping was performed on the Wikipedia page of each continental competition to obtain team names, their confederations, and countries.
- CAF: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CAF_countries
- CONMEBOL: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CONMEBOL_countries
- UEFA: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_UEFA_countries
- CONCACAF: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CONCACAF_countries
- AFC: https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_AFC_countries

## 1.2. Global Competitions

### 1.2.1. Club World Cup:
- 2025/2025: https://footystats.org/c-dl.php?type=matches&comp=13878
- 2024/2024: https://footystats.org/c-dl.php?type=matches&comp=13557
- 2023/2023: https://footystats.org/c-dl.php?type=matches&comp=10958
- 2022/2022: https://footystats.org/c-dl.php?type=matches&comp=8830
- 2021/2021: https://footystats.org/c-dl.php?type=matches&comp=7069
- 2020/2020: https://footystats.org/c-dl.php?type=matches&comp=5517
- 2019/2019: https://footystats.org/c-dl.php?type=matches&comp=3370
- 2018/2018: https://footystats.org/c-dl.php?type=matches&comp=1813

### 1.2.2. World Cup:
- https://fbref.com/en/comps/1/

## 1.3. Players and Matches
Transfermarkt: https://www.kaggle.com/datasets/davidcariboo/player-scores
*( From this dataset, the following tables were used: `appearances.csv`, `players.csv`, `clubs.csv`, `competitions.csv`, and `games.csv` )*

# 2. DATA USAGE AND PIPELINE

The main goal of this section is to structure and quantify player performance on a global scale to feed a Machine Learning model capable of evaluating the true strength of National Teams. To ensure fair metrics, the data pipeline was designed in three layers (Medallion Architecture):

## 2.1. Raw Layer:
Extraction and storage of the original data from the cited sources, without modifications.

## 2.2. Processed Layer:
1. Confederation Weights Calculation: Uses the head-to-head history from the Club World Cup to calculate the relative strength of each continent (e.g., UEFA, CONMEBOL, CAF) based on Points Per Game (PPG).
2. Relational Mapping: Merges Transfermarkt tables (clubs and competitions) to dynamically assign the correct confederation to each club and stamp the calculated difficulty weight onto each player.

## 2.3. Refined Layer:
Generates the final datasets for Machine Learning focused on two universal evaluation fronts (Win Rate and Plus/Minus), allowing for a fair comparison between attackers and defenders:
1. Individual Efficiency: Evaluates the isolated impact of each player by multiplying their wins and plus/minus by the difficulty weight of their league.
2. Synergy Matrix (Graphs): Maps pairs of players of the same nationality who play together at the same club. It measures the "chemistry" of the partnership at the club level to predict the synergy they will bring together to the National Team.