import pandas as pd
import pytest
from movie_recommendations import clean_movie_title, find_similar_movies, search

similar_movies_test = pd.read_csv('Python_Essentials_for_Mlops/Project_01/data_test/similar_movies.csv')
search_movies_test = pd.read_csv('Python_Essentials_for_Mlops/Project_01/data_test/search_movies.csv')


# Teste para a função find_similar_movies
def test_find_similar_movies():
    result = find_similar_movies(5)
    expected_result = similar_movies_test
    assert(result, expected_result)

# Teste para a função search
def test_search():
    expected_result = search_movies_test
    result = search('Toy Story')
    assert(result, expected_result)
