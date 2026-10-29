"""Offline text translation with Meta's NLLB-200 (600M, CTranslate2 int8). Runs on the GPU when it can."""
import re
import threading

import stt  # noqa: F401,I001  (first: sets CUDA DLL folders and download settings)

import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
LANG_CODES = {"en": "eng_Latn", "es": "spa_Latn", "ms": "zsm_Latn"}
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


class Translator:
    def __init__(self, log=print):
        patterns = ["*.bin", "*.json", "*.txt", "*.model"]
        try:  # downloaded copy first, so start-up does not wait for the internet
            path = snapshot_download(MODEL_REPO, allow_patterns=patterns, local_files_only=True)
        except Exception:
            path = snapshot_download(MODEL_REPO, allow_patterns=patterns)
        self.sp = spm.SentencePieceProcessor(model_file=f"{path}/sentencepiece.bpe.model")
        self.lock = threading.Lock()
        try:
            self.model = ctranslate2.Translator(path, device="cuda", compute_type="int8_float16")
            self._translate_one("hello", "en", "es")  # surface GPU errors now
            self.device = "cuda"
        except Exception as e:
            log(f"Translator: GPU not usable ({str(e).splitlines()[0]}). Using CPU.")
            self.model = ctranslate2.Translator(path, device="cpu", compute_type="int8")
            self.device = "cpu"

    def _translate_one(self, text, source, target):
        return self._translate_sentences([text], source, target)[0]

    def _translate_sentences(self, sentences, source, target):
        src = LANG_CODES[source]
        tgt = LANG_CODES[target]
        batch = [[src] + self.sp.encode(s, out_type=str) + ["</s>"] for s in sentences]
        with self.lock:
            results = self.model.translate_batch(batch, target_prefix=[[tgt]] * len(batch),
                                                 beam_size=2, max_decoding_length=256)
        texts = [self.sp.decode([t for t in r.hypotheses[0] if t != tgt]).strip() for r in results]
        return [t[2:] if t.startswith("- ") else t for t in texts]  # NLLB sometimes adds a dialogue dash

    def translate(self, text, source, target):
        # NLLB translates one sentence at a time and drops the rest, so split first.
        sentences = [s for s in SENTENCE_END.split(text.strip()) if s.strip()]
        if not sentences:
            return ""
        return " ".join(self._translate_sentences(sentences, source, target))
