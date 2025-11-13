# aws_services/transcribe.py

import asyncio
import logging
import time
from typing import AsyncGenerator, Callable, Optional

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TranscriptionHandler(TranscriptResultStreamHandler):
    """
    Handle AWS Transcribe streaming events.
    Builds up the final transcript and invokes a callback for every chunk.
    """

    def __init__(self,
                 output_stream,
                 transcript_callback: Callable[[str, float], asyncio.Future]):
        """
        :param output_stream: the output stream from Transcribe.
        :param transcript_callback: async function called when a partial or final transcript arrives.
                                    receives (transcript_str, timestamp).
        """
        super().__init__(output_stream)
        self.transcript_callback = transcript_callback
        self.final_transcript: str = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            # skip partial results if you only care about final
            if result.is_partial:
                continue
            for alt in result.alternatives:
                text = alt.transcript
                self.final_transcript += text + " "
                timestamp = time.time()
                # invoke callback with the chunk text & timestamp
                await self.transcript_callback(text, timestamp)

async def start_transcription(audio_stream: AsyncGenerator[bytes, None],
                              transcript_callback: Callable[[str, float], asyncio.Future],
                              region: str = "ap-south-1",
                              language_code: str = "en-US",
                              media_sample_rate_hz: int = 16000,
                              media_encoding: str = "pcm") -> str:
    """
    Start AWS Transcribe streaming session.

    :param audio_stream: Async generator yielding raw audio chunks (bytes).
    :param transcript_callback: Async callback for each completed chunk of text.
    :param region: AWS region.
    :param language_code: language of audio.
    :param media_sample_rate_hz: sample rate of audio stream.
    :param media_encoding: encoding of audio.
    :returns: full final transcript (string).
    """
    try:
        client = TranscribeStreamingClient(region=region)
        logger.info(f"Starting transcription in region {region} with language {language_code}")
        stream = await client.start_stream_transcription(
            language_code=language_code,
            media_sample_rate_hz=media_sample_rate_hz,
            media_encoding=media_encoding,
            show_speaker_label=False  # set true if you want speaker diarization
        )

        handler = TranscriptionHandler(stream.output_stream, transcript_callback)

        async def write_chunks():
            async for chunk in audio_stream:
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await stream.input_stream.end_stream()
            logger.info("Finished sending audio stream")

        write_task = asyncio.create_task(write_chunks())
        handle_task = asyncio.create_task(handler.handle_events())

        await asyncio.gather(write_task, handle_task)

        final_text = handler.final_transcript.strip()
        logger.info(f"Final transcript: {final_text}")
        return final_text

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return ""
