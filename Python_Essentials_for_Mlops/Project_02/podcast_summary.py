"""
Pipeline Airflow

This script provides a simple pipeline airflow.
"""
import os
import json
import logging
import requests
import xmltodict
import pendulum
from airflow.decorators import dag, task
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

logging.basicConfig(filename='movie_recommendation.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# URL do podcast e pasta para armazenar os episódios baixados
PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "episodes"
FRAME_RATE = 16000

@dag(
    dag_id='podcast_summary',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)
def podcast_summary():
    """
    DAG (Directed Acyclic Graph) para resumir um podcast, incluindo o download
    de episódios, transcrição de áudio e armazenamento em um banco de dados SQLite.
    """
    create_database_task()
    podcast_episodes = get_episodes_task()
    load_episodes_task(podcast_episodes)
    download_episodes_task(podcast_episodes)
    speech_to_text_task()

def create_database_task():
    """
    Cria a tabela SQLite para armazenar informações de episódios do podcast.
    """
    return SqliteOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )

@task()
def get_episodes_task():
    """
    Obtém a lista de episódios do feed do podcast.
    """
    try:
        data = requests.get(
            PODCAST_URL, timeout=10)
        data.raise_for_status()
        feed = xmltodict.parse(data.text)
        episodes = feed["rss"]["channel"]["item"]
        print(f"Found {len(episodes)} episodes.")
        return episodes
    except requests.exceptions.RequestException as get_episodes_task_execptions:
        logging.error("Error while fetching podcast episodes %s:",
                      str(get_episodes_task_execptions))
        return []

@task()
def load_episodes_task(episodes):
    """
    Carrega novos episódios no banco de dados.
    """
    hook = SqliteHook(sqlite_conn_id="podcasts")
    stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
    new_episodes = []

    for episode in episodes:
        link = episode["link"]
        if link not in stored_episodes["link"].values:
            filename = f"{link.split('/')[-1]}.mp3"
            new_episodes.append([link, episode["title"],
                                episode["pubDate"], episode["description"], filename])

    hook.insert_rows(table='episodes', rows=new_episodes, target_fields=[
                     "link", "title", "published", "description", "filename"])
    return new_episodes

@task()
def download_episodes_task(episodes):
    """
    Baixa os episódios de áudio do podcast.
    """
    audio_files = []
    for episode in episodes:
        link = episode["link"]
        name_end = link.split('/')[-1]
        filename = f"{name_end}.mp3"
        audio_path = os.path.join(EPISODE_FOLDER, filename)

        if not os.path.exists(audio_path):
            print(f"Downloading {filename}")
            try:
                audio = requests.get(
                    episode["enclosure"]["@url"], timeout=10)
                audio.raise_for_status()
                with open(audio_path, "wb+") as f:
                    f.write(audio.content)
            except requests.exceptions.RequestException as download_episodes_exceptions:
                logging.error("Error while downloading audio %s:",
                              str(download_episodes_exceptions))
        audio_files.append({"link": link, "filename": filename})
    return audio_files

@task()
def speech_to_text_task():
    """
    Realiza a transcrição de áudio dos episódios baixados.
    """
    hook = SqliteHook(sqlite_conn_id="podcasts")
    untranscribed_episodes = hook.get_pandas_df(
        "SELECT * from episodes WHERE transcript IS NULL;")

    model = Model(model_name="vosk-model-en-us-0.22-lgraph")
    rec = KaldiRecognizer(model, FRAME_RATE)
    rec.SetWords(True)

    for _, row in untranscribed_episodes.iterrows():
        print(f"Transcribing {row['filename']}")
        filepath = os.path.join(EPISODE_FOLDER, row["filename"])
        mp3 = AudioSegment.from_mp3(filepath)
        mp3 = mp3.set_channels(1)
        mp3 = mp3.set_frame_rate(FRAME_RATE)

        step = 20000
        transcript = ""
        for i in range(0, len(mp3), step):
            print(f"Progress: {i/len(mp3)}")
            segment = mp3[i:i+step]
            rec.AcceptWaveform(segment.raw_data)
            result = rec.Result()
            text = json.loads(result)["text"]
            transcript += text

        hook.insert_rows(table='episodes', rows=[
                         [row["link"], transcript]],
                         target_fields=["link", "transcript"], replace=True)

if __name__ == "__main__":
    SUMMARY = podcast_summary()
