# Live Translator

Talk with Spanish or Malay speakers when you only speak English. Pick the language in **Their language**.

- **They speak Spanish / Malay** → you see their words and the English in the window.
- **You speak English** → they hear a Spanish / Malay voice through a virtual microphone.

## Start

Double-click `run.bat`. On the first run it creates `.venv` and installs everything (a few minutes). After that it just launches the app.

The first start also downloads the speech model and the translation model (about 2 GB together), so wait a few minutes that first time.

## Setup in Zoom / Google Meet / Discord / WhatsApp

1. In the app, set **Send Spanish voice to** = `CABLE Input (VB-Audio Virtual Cable)`.
2. In Zoom (or Meet), set **Microphone** = `CABLE Output (VB-Audio Virtual Cable)`.
3. In the app, set **Listen to** = the speakers or headphones where you hear the call.
4. Click **Start listening**.
5. Use **headphones** (so the app doesn't pick up its own output).

You need [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) installed for the virtual microphone.

## How to talk

- Hold **F8** (works even when Zoom is in front) or hold the green button, speak English, then release.
- Or type English in the box at the bottom and press Enter.

Wait for the translated voice to finish before you talk again. Each sentence takes about 1–3 seconds.

## How it works

| Stage | Component | Notes |
| --- | --- | --- |
| Speech → text | `stt.py` (faster-whisper) | Runs locally on the GPU. |
| Translate | `translate.py` (NLLB) and `llm_translate.py` (SEA-LION via Ollama) | See engines below. |
| Text → voice | `tts.py` (Microsoft Edge TTS) | Needs internet; this is the only online part. |
| Audio devices | `audio_io.py` | Loopback capture and virtual-mic output. |
| UI | `app.py` (Tkinter) | Settings saved to `settings.json` on close. |

### Translation engines

- **Natural (local AI)** — the SEA-LION model running in Ollama. Sounds like real spoken Malay/Spanish. Ollama must be installed; it starts on its own.
- **Basic (offline NLLB)** — more formal, fully offline. Used automatically if the local AI isn't available.

## Dev setup

Requires Python 3.11 and an NVIDIA GPU (CUDA). `run.bat` handles the venv, or do it by hand:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\pythonw.exe app.py
```

Dependencies are in `requirements.txt` (faster-whisper, pyaudiowpatch, sounddevice, edge-tts, keyboard, CUDA runtime wheels, and others).

## Notes

- If the speech model is slow, choose `small` in **Speech model**.
- Settings (devices, voices, language, engine) are saved in `settings.json` when you close the app.
- `app.log` holds the last run's log for troubleshooting.
