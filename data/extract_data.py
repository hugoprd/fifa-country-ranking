import soccerdata as sd

# Você pode especificar as competições continentais (ex: Copa Libertadores e Champions League)
# e o intervalo de anos (2002 a 2026)
fbref = sd.FBref(leagues=['Copa Libertadores', 'Champions League'], seasons=range(2002, 2027))

# Extrai o histórico de partidas com os resultados
match_history = fbref.read_match_results()
