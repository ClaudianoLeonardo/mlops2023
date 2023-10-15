
# Movie Recommendation System

This Python script provides a simple movie recommendation system based on user input. It uses a dataset of movie titles and user ratings to suggest similar movies to the one entered by the user.

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Data](#data)
4. [Functions](#functions)
5. [Usage](#usage)
6. [Logging](#logging)
7. [License](#license)
8. [References](#references)

## Introduction

The Movie Recommendation System is a tool that allows users to input a movie title, and it will provide recommendations for other movies that are similar based on user ratings. It utilizes the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and [cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) to calculate similarity between movies based on their titles and user ratings.

## Dependencies

To run this code, you need the following Python libraries:

- `logging`: For error logging and debugging.
- `re`: For regular expression-based text cleaning.
- `numpy` and `pandas`: For data manipulation and analysis.
- `ipywidgets` and `IPython.display`: For creating interactive widgets in Jupyter Notebook or JupyterLab.
- `scikit-learn`: For text vectorization and similarity calculations.

You can install these dependencies using `pip` or any other package manager.

```bash
pip install numpy pandas ipywidgets scikit-learn
```

## Data

This script relies on two data files:

- `movies.csv`: This file contains information about movies, including their titles and genres.
- `ratings.csv`: This file contains user ratings for these movies.

Please make sure to provide the correct file paths to these datasets before running the script. The script will load and process the data during execution.

## Functions

### `clean_movie_title(title)`

This function removes non-alphanumeric characters from a movie title.

### `find_similar_movies(movie_id)`

Given a movie ID, this function finds similar movies based on user ratings and returns a DataFrame with the top recommendations.

### `on_type_recommendations(data)`

This function updates the list of recommendations based on user input in the movie title widget.

### `search(title)`

This function performs a movie search based on the title provided, cleans the title, and calculates movie similarity using TF-IDF and cosine similarity.

## Usage

1. Import the necessary libraries and data files.
2. Create an interactive widget for inputting a movie title.
3. Enter a movie title, and the script will provide movie recommendations based on user ratings.

## Logging

The script logs errors and exceptions to a file named `movie_recommendation.log` using the `logging` library. This can be useful for debugging and troubleshooting issues.

## License

This project is provided under the [MIT License](LICENSE) for open-source use and modification. Please review the license for more details.

Enjoy using the Movie Recommendation System!

## References

[dataquest](https://www.dataquest.io/)


[mlops](https://github.com/ivanovitchm/mlops)
