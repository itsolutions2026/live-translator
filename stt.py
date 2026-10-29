"""Speech-to-text with faster-whisper. Uses the GPU when it works, otherwise the CPU."""
import os
import site
import threading

import numpy as np

# The hf_xet download backend crashes the process on this machine; use plain HTTP downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Text Whisper often invents from noise or silence.
HALLUCINATIONS = {
    "subtítulos realizados por la comunidad de amara.org",
    "gracias por ver el video", "gracias por ver el vídeo", "gracias por ver",
    "suscríbete", "¡suscríbete!", "thank you.", "thanks for watching!",
    "thank you for watching.", "you", "...", ".",
}


def _add_cuda_dll_paths():
    """Let CTranslate2 find the cuBLAS / cuDNN DLLs installed by pip (nvidia-* packages)."""
    roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in roots:
        for lib in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            d = os.path.join(root, "nvidia", lib, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


_add_cuda_dll_paths()
from faster_whisper import WhisperModel  # noqa: E402  (must come after the DLL paths)


class Transcriber:
    def __init__(self, model_name, log=print):
        self.lock = threading.Lock()
        try:
            self.model = self._load(model_name, "cuda", "float16")
            self._warm_up()
            self.device = "cuda"
        except Exception as e:
            log(f"GPU not usable ({str(e).splitlines()[0]}). Using CPU, it will be slower.")
            self.model = self._load(model_name, "cpu", "int8")
            self.device = "cpu"

    @staticmethod
    def _load(model_name, device, compute_type):
        # Use the downloaded copy first, so start-up does not wait for the internet.
        try:
            return WhisperModel(model_name, device=device, compute_type=compute_type, local_files_only=True)
        except Exception:
            return WhisperModel(model_name, device=device, compute_type=compute_type)

    def _warm_up(self):
        # Runs the model once so GPU errors show up now, not in the middle of a call.
        segments, _ = self.model.transcribe(np.zeros(16000, np.float32), language="en", beam_size=1)
        list(segments)

    def transcribe(self, audio, language, vad=True):
        # vad=False for push-to-talk: the key is held, so it is speech, and the voice detector
        # would throw away quiet speech.
        with self.lock:
            segments, _ = self.model.transcribe(
                audio, language=language, beam_size=1, vad_filter=vad,
                condition_on_previous_text=False, without_timestamps=True)
            parts = [s.text.strip() for s in segments
                     if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
        text = " ".join(p for p in parts if p).strip()
        return "" if text.lower() in HALLUCINATIONS else text
