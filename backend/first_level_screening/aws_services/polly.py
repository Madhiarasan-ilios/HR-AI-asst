# aws_services/polly.py

import boto3
import logging
from typing import BinaryIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PollyClient:
    def __init__(self, region_name: str = "ap-south-1", voice_id: str = "Joanna", engine: str = "neural"):
        """
        Initialize the Polly client.
        :param region_name: AWS region.
        :param voice_id: The Polly voice to use.
        :param engine: Either 'standard' or 'neural' (neural gives higher quality).
        """
        self.region_name = region_name
        self.voice_id = voice_id
        self.engine = engine
        self.client = boto3.client("polly", region_name=region_name)
        logger.info(f"Initialized Polly client in region {region_name} using voice {voice_id} and engine {engine}")

    def synthesize_to_stream(self, text: str, output_format: str = "mp3") -> BinaryIO:
        """
        Synthesizes speech and returns the audio stream.
        :param text: Text to speak.
        :param output_format: Format of the audio ('mp3', 'pcm', 'ogg_vorbis').
        :return: Audio stream (file-like object).
        """
        try:
            response = self.client.synthesize_speech(
                Text=text,
                VoiceId=self.voice_id,
                OutputFormat=output_format,
                Engine=self.engine
            )
            audio_stream = response.get("AudioStream")
            logger.info("Polly synthesis succeeded")
            return audio_stream
        except Exception as e:
            logger.error(f"Polly synthesize_speech error: {e}")
            raise

    def save_to_file(self, text: str, filepath: str, output_format: str = "mp3"):
        """
        Synthesizes speech and saves to a file.
        :param text: Text to speak.
        :param filepath: Path to save the audio file.
        :param output_format: Format of the audio.
        """
        stream = self.synthesize_to_stream(text=text, output_format=output_format)
        with open(filepath, "wb") as f:
            f.write(stream.read())
        logger.info(f"Saved audio file to {filepath}")
