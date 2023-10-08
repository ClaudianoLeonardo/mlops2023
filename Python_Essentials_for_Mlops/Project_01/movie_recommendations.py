"""
Movie Recommendation System

This script provides a simple movie recommendation system based on user input.
"""
import re
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_movie_title(title):
    """
    Remove caracteres não alfanuméricos do título do filme.

    Args:
        title (str): O título do filme.

    Returns:
        str: O título do filme após a limpeza.
    """
    return re.sub("[^a-zA-Z0-9 ]", "", title)

def find_similar_movies(movie_id):
    """
    Encontra filmes similares com base nas avaliações de usuários.

    Args:
        movie_id (int): O ID do filme para o qual deseja encontrar recomendações.

    Returns:
        pandas.DataFrame: Um DataFrame com as principais recomendações de filmes.
    """

    similar_users = ratings[(ratings["movieId"] == movie_id) &
                            (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
                                (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) &
                        (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return (
    rec_percentages.head(10)
    .merge(movies, left_index=True, right_on="movieId")
    [["score", "title", "genres"]]
)


def on_type_recommendations(data):
    """
    Atualiza a lista de recomendações com base na entrada do usuário.

    Args:
        data (ipywidgets.WidgetEvent): O evento de mudança de valor do widget de entrada de filme.
    """
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

def search(title):
    """
    Realiza uma pesquisa de filmes com base no título fornecido.

    Args:
        title (str): O título do filme a ser pesquisado.

    Returns:
        pandas.DataFrame: Um DataFrame com os resultados da pesquisa.
    """
    title = clean_movie_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

# Carregar dados
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Configurar widget de entrada de título de filme
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

# Observar a entrada do usuário para atualizar a lista de recomendações
movie_name_input.observe(on_type_recommendations, names='value')

# Exibir o widget de entrada e a lista de recomendações
display(movie_name_input, recommendation_list)

# Configurar vetorizador TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Calcular TF-IDF para os títulos de filmes
tfidf = vectorizer.fit_transform(movies["title"])
