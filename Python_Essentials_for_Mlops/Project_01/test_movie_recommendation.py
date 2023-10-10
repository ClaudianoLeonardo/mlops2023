import pandas as pd
import pytest
from movie_recommendations import clean_movie_title, find_similar_movies, search

# Crie dataframes de teste
# Suponha que seus dataframes de teste tenham nomes e tipos de dados compatíveis com suas funções

test_movies = pd.DataFrame({
    "movieId": [1, 2, 3],
    "title": ["The Dark Knight (2008)", "Inception", "Toy Story (1995)"],
    "genres": ["Action", "Sci-Fi", "Animation"]
})

test_ratings = pd.DataFrame({
    "userId": [1, 2, 3],
    "movieId": [1, 2, 3],
    "rating": [5, 4, 5]
})

# Teste para a função clean_movie_title
def test_clean_movie_title():
    # Teste com um dataframe de exemplo
    input_title = "The Dark Knight (2008)"
    expected_result = "The Dark Knight 2008"
    result = clean_movie_title(input_title)
    assert result == expected_result

# Teste para a função find_similar_movies
def test_find_similar_movies():
    # Suponha que você tenha dados de teste para movies e ratings
    # e crie um ambiente controlado para testar a função
    expected_result = pd.DataFrame({
        "score": [1.0, 0.5, 0.25],
        "title": ["The Dark Knight (2008)", "Inception", "Toy Story (1995)"],
        "genres": ["Action", "Sci-Fi", "Animation"]
    })
    
    # Chame a função com os dataframes de teste
    result = find_similar_movies(1, movies=test_movies, ratings=test_ratings)
    
    # Use uma função de comparação de dataframes para verificar os resultados
    pd.testing.assert_frame_equal(result, expected_result)

# Teste para a função search
def test_search():
    # Suponha que você tenha dados de teste para movies
    # e crie um ambiente controlado para testar a função
    expected_result = pd.DataFrame({
        "title": ["The Dark Knight (2008)", "Inception", "Toy Story (1995)"],
        "genres": ["Action", "Sci-Fi", "Animation"]
    })
    
    # Chame a função com os dataframes de teste
    result = search("The Dark Knight", movies=test_movies)
    
    # Use uma função de comparação de dataframes para verificar os resultados
    pd.testing.assert_frame_equal(result, expected_result)
