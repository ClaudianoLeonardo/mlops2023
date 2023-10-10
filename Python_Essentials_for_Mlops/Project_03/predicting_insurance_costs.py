"""
Predicting Insurance Costs

Este script carrega dados de seguros de saúde, treina um modelo de regressão linear
para prever os custos dos seguros e avalia o desempenho do modelo.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Carrega os dados do arquivo CSV especificado.

    Args:
        file_path (str): O caminho do arquivo CSV a ser carregado.

    Returns:
        pandas.DataFrame: O DataFrame contendo os dados do arquivo CSV.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError as file_not_found:
        logger.error("Arquivo não encontrado: Verifique o caminho do arquivo. Erro : %s",
                     str(file_not_found))
        return None
    except pd.errors.EmptyDataError as empty_data:
        logger.error("Arquivo CSV está vazio. Erro: %s", str(empty_data))
        return None
    except pd.errors.ParserError as parser_error:
        logger.error("Erro ao analisar o arquivo CSV. Erro: %s", str(parser_error))
        return None

def transform_data(data):
    """
    Realiza transformações nos dados, calculando o logaritmo base 2 das "charges" e
    criando uma coluna booleana "is_smoker" com base na coluna "smoker".

    Args:
        data (pandas.DataFrame): O DataFrame contendo os dados originais.

    Returns:
        tuple: Um par de DataFrames contendo as características (X) e os alvos (y)
        transformados.
    """
    try:
        data["log_charges"] = np.log2(data["charges"])
        data["is_smoker"] = data["smoker"] == "yes"
        return data[["age", "bmi", "is_smoker"]], data["log_charges"]
    except KeyError as key_error:
        logger.error("Colunas necessárias não encontradas nos dados. Erro: %s", str(key_error))
        return None

def train_linear_regression(data_train, labels_train):
    """
    Treina um modelo de regressão linear com base nos dados de treinamento.

    Args:
        data_train (pandas.DataFrame): As características de treinamento.
        labels_train (pandas.Series): Os alvos de treinamento.

    Returns:
        sklearn.linear_model.LinearRegression: O modelo treinado.
    """
    try:
        model = LinearRegression()
        model.fit(data_train, labels_train)
        return model
    except ValueError as value_error:
        logger.error("Erro ao treinar o modelo de regressão linear. Erro: %s", str(value_error))
        return None

def evaluate_model(model, data, labels):
    """
    Avalia o desempenho de um modelo usando a métrica de erro quadrático médio (MSE)
    e o coeficiente de determinação (R^2).

    Args:
        model (sklearn.linear_model.LinearRegression): O modelo a ser avaliado.
        data (pandas.DataFrame): As características de entrada.
        labels (pandas.Series): Os alvos verdadeiros.

    Returns:
        tuple: Um par de valores MSE e R^2.
    """
    try:
        labels_pred = model.predict(data)
        mse = mean_squared_error(labels, labels_pred)
        r2 = r2_score(labels, labels_pred)
        return mse, r2
    except ValueError as value_error:
        logger.error("Erro ao avaliar o modelo. Erro: %s", str(value_error))
        return None

def main():
    """
    Função principal que carrega os dados, treina o modelo, e avalia o desempenho
    do modelo nos dados de treinamento e teste.
    """
    try:
        file_path = "insurance.csv"
        insurance_data = load_data(file_path)
        data, labels = transform_data(insurance_data)

        data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                            test_size=0.25, random_state=1)

        insurance_model = train_linear_regression(data_train, labels_train)

        train_mse, train_r2 = evaluate_model(insurance_model, data_train, labels_train)
        test_mse, test_r2 = evaluate_model(insurance_model, data_test, labels_test)
        return(
        print("Training MSE:", train_mse),
        print("Training R^2:", train_r2),
        print("Test MSE:", test_mse),
        print("Test R^2:", test_r2)
        )

    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            KeyError, ValueError) as exec_error:
        logger.error("Erro durante a execução do programa: %s", str(exec_error))
        return None

if __name__ == "__main__":
    main()
