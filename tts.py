"""Text-to-speech with Microsoft Edge neural voices (free, needs internet)."""
import asyncio
import time

import edge_tts
import edge_tts.communicate
import miniaudio
import numpy as np
from edge_tts.constants import TRUSTED_CLIENT_TOKEN

# Since October 2026 Microsoft refuses speech on the old host (403). Edge now uses this one.
# Same fix as edge-tts PR #491; remove once a release with it is installed.
if "speech.platform.bing.com" in edge_tts.communicate.WSS_URL:
    edge_tts.communicate.WSS_URL = ("wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1"
                                    f"?Ocp-Apim-Subscription-Key={TRUSTED_CLIENT_TOKEN}")

VOICES = {  # language code -> {label: Edge voice id}
    "es": {
        "Jorge (Mexico, male)": "es-MX-JorgeNeural",
        "Dalia (Mexico, female)": "es-MX-DaliaNeural",
        "Alonso (US Spanish, male)": "es-US-AlonsoNeural",
        "Paloma (US Spanish, female)": "es-US-PalomaNeural",
        "Alvaro (Spain, male)": "es-ES-AlvaroNeural",
        "Elvira (Spain, female)": "es-ES-ElviraNeural",
        "Gonzalo (Colombia, male)": "es-CO-GonzaloNeural",
        "Salome (Colombia, female)": "es-CO-SalomeNeural",
        "Tomas (Argentina, male)": "es-AR-TomasNeural",
        "Elena (Argentina, female)": "es-AR-ElenaNeural",
    },
    "ms": {
        "Osman (Malaysia, male)": "ms-MY-OsmanNeural",
        "Yasmin (Malaysia, female)": "ms-MY-YasminNeural",
    },
}

SAMPLE_RATE = 24000


def synthesize(text, voice, attempts=3):
    """Returns (float32 mono audio, sample_rate)."""
    async def fetch():
        mp3 = bytearray()
        # Short timeouts + retries: over a VPN one connection often stalls while the next works.
        async for chunk in edge_tts.Communicate(text, voice, connect_timeout=5, receive_timeout=15).stream():
            if chunk["type"] == "audio":
                mp3.extend(chunk["data"])
        if not mp3:
            raise RuntimeError("the voice service sent no audio")
        return bytes(mp3)

    for attempt in range(attempts):
        try:
            mp3 = asyncio.run(fetch())
            break
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)
    decoded = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.FLOAT32,
                               nchannels=1, sample_rate=SAMPLE_RATE)
    return np.asarray(decoded.samples, dtype=np.float32), SAMPLE_RATE
