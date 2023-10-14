import os
import pytest
from podcast_summary import speech_to_text_task


# Mock SqliteOperator
class MockSqliteOperator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def execute(self, context):
        pass


def test_speech_to_text_task():
    # Mock SqliteHook and untranscribed episodes
    hook = MockSqliteOperator()
    untranscribed_episodes = [{"link": "https://example.com/episode1", "filename": "episode1.mp3"}]
    hook.get_pandas_df = lambda query: untranscribed_episodes

    # Mock vosk.Model and KaldiRecognizer
    class MockModel:
        def __init__(self, model_name):
            pass

    class MockKaldiRecognizer:
        def __init__(self, model, frame_rate):
            self.transcript = ""

        def SetWords(self, value):
            pass

        def AcceptWaveform(self, data):
            self.transcript += "Mock Transcript "

        def Result(self):
            return '{"text": "' + self.transcript.strip() + '"}'

    with pytest.raises(Exception):
        speech_to_text_task()

