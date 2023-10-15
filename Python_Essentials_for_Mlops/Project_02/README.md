# Podcast Summary Airflow Pipeline

This repository contains an Airflow DAG (Directed Acyclic Graph) for summarizing a podcast. The pipeline includes downloading podcast episodes, transcribing audio content, and storing episode information in a SQLite database.

## Prerequisites

Before running this Airflow pipeline, make sure you have the following dependencies installed:

- Python 3.x
- [Apache Airflow](https://airflow.apache.org/)
- [Vosk](https://github.com/alphacep/vosk-api) speech recognition library
- [pydub](https://github.com/jiaaro/pydub) for audio processing
- SQLite database
- Internet access for downloading podcast episodes

## Configuration

You can configure the pipeline by modifying the following variables in the script:

- `PODCAST_URL`: The URL of the podcast RSS feed you want to summarize.
- `EPISODE_FOLDER`: The folder where downloaded episodes will be stored.
- `FRAME_RATE`: The audio frame rate for processing.

## DAG Structure

The Airflow DAG `podcast_summary` is defined with the following tasks:

1. **create_database_task**: This task creates a SQLite table to store episode information.

2. **get_episodes_task**: This task fetches a list of episodes from the podcast's RSS feed.

3. **load_episodes_task**: This task loads new episodes into the database. It checks for episodes that haven't been stored and inserts them into the database.

4. **download_episodes_task**: This task downloads audio files of the podcast episodes. It checks if the episode has already been downloaded to avoid duplicates.

5. **speech_to_text_task**: This task performs audio transcription for downloaded episodes. It uses the Vosk speech recognition library to transcribe audio content and stores the transcript in the database.

## Usage

1. Ensure you have Apache Airflow set up and configured on your system.

2. Install the required Python packages using `pip install vosk pydub pendulum requests xmltodict`.

3. Configure the variables (`PODCAST_URL`, `EPISODE_FOLDER`, and `FRAME_RATE`) in the script to match your requirements.

4. Save the script and run it to create the Airflow DAG.

5. Start the Airflow scheduler and web server to monitor and trigger the pipeline. Use the Airflow web interface to trigger the `podcast_summary` DAG and view its progress.

6. The pipeline will download, transcribe, and store podcast episode information in the SQLite database. You can customize the pipeline's schedule and other settings as needed.

## Logging

Log files are created in the current working directory with the filename "movie_recommendation.log" for error messages and status updates during the pipeline's execution.

## License

This project is provided under the [MIT License](LICENSE) for open-source use and modification. Please review the license for more details.

## References

[dataquest](https://www.dataquest.io/)


[mlops](https://github.com/ivanovitchm/mlops)
