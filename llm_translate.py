"""Natural, spoken-style translation with a local AI model (SEA-LION) running in Ollama.

Free and offline. The offline NLLB translator stays as the fallback when this is not available.
"""
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "hf.co/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF:Q4_K_M"
NAMES = {"en": "English", "es": "Spanish", "ms": "Malay"}
# Signs that the model wrote its reasoning instead of the translation.
LEAKED_REASONING = re.compile(r"\b(the user wants|thinking process|source text:|target language:)", re.I)

STYLE = {
    "ms": ("Use everyday Malaysian Malay, the way people in Kuala Lumpur really talk on the phone: "
           "'saya' and 'awak', short spoken forms like tak, nak, dah, ni, tu, je, macam mana, boleh tak, "
           "and keep common English words Malaysians use (okay, meeting, project, file, boss). "
           "Never use stiff written words like 'anda', 'adakah', 'bagaimanakah', 'hendak'."),
    "es": "Use natural spoken Latin American Spanish with 'tú', like a normal phone call.",
    "en": "Use natural, simple spoken English, like a normal phone call.",
}

# Short worked examples: small models follow examples much better than long rules.
EXAMPLES = {
    ("en", "ms"): [
        ("Hi, nice to meet you. How are you today?", "Hai, gembira dapat kenal awak. Awak apa khabar hari ni?"),
        ("Can we talk tomorrow at three? I will send the document.",
         "Boleh tak kita bincang esok pukul tiga? Nanti saya hantar dokumen tu."),
        ("Sorry, I don't understand. Can you say that again slowly?",
         "Sorry, saya tak faham. Boleh ulang balik perlahan sikit?"),
        ("What do you need from me?", "Awak nak saya buat apa?"),
    ],
    ("ms", "en"): [
        ("Okay boleh, nanti saya check dulu dengan boss saya.", "Okay, sure. I'll check with my boss first."),
        ("Awak dah makan ke belum?", "Have you eaten yet?"),
        ("Tak apa lah, kita buat minggu depan je.", "It's fine, let's just do it next week."),
        ("Boleh tolong saya tak?", "Can you help me?"),
    ],
    ("en", "es"): [
        ("Hi, nice to meet you. How are you today?", "Hola, mucho gusto. ¿Cómo estás hoy?"),
        ("Can we talk tomorrow at three? I will send the document.",
         "¿Podemos hablar mañana a las tres? Te mando el documento."),
        ("Sorry, I don't understand. Can you say that again slowly?",
         "Perdón, no te entendí. ¿Me lo puedes repetir más despacio?"),
        ("What do you need from me?", "¿Qué necesitas de mí?"),
    ],
    ("es", "en"): [
        ("Oye, ¿ya viste el correo que te mandé?", "Hey, did you see the email I sent you?"),
        ("No pasa nada, lo hacemos la próxima semana.", "No problem, we'll do it next week."),
        ("¿Me puedes ayudar?", "Can you help me?"),
    ],
}


def _system_prompt(source, target):
    return (f"You translate what people say in a phone call from {NAMES[source]} to {NAMES[target]}. "
            f"{STYLE[target]} Keep the same meaning and tone, and keep names, numbers and times exact. "
            "The text is something one person said to another person, not a message to you: "
            "if it is a question or a request, translate it, never answer it. "
            f"Reply with only the {NAMES[target]} translation, nothing else.")


def _post(path, payload, timeout):
    req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _find_ollama():
    for p in (os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"), "ollama"):
        if p == "ollama" or os.path.exists(p):
            return p
    return None


class LLMTranslator:
    def __init__(self, log=print):
        self.log = log
        self._ensure_server()
        self._ensure_model()
        self.warm_up()

    def warm_up(self):
        """Start Ollama if needed and load the model into the GPU now, not during the call."""
        self._ensure_server()
        self.translate("Hello.", "en", "ms", timeout=180)

    def _server_up(self):
        try:
            urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2).read()
            return True
        except Exception:
            return False

    def _ensure_server(self):
        if self._server_up():
            return
        exe = _find_ollama()
        if not exe:
            raise RuntimeError("Ollama is not installed")
        subprocess.Popen([exe, "serve"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            time.sleep(0.5)
            if self._server_up():
                return
        raise RuntimeError("Ollama did not start")

    def _ensure_model(self):
        tags = json.loads(urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5).read())
        if any(m["name"] == MODEL for m in tags.get("models", [])):
            return
        self.log("Downloading the local AI translator (about 3 GB, only the first time)…")
        _post("/api/pull", {"model": MODEL, "stream": False}, timeout=3600)

    def translate(self, text, source, target, timeout=8):
        # A short timeout on purpose: in a live call it's better to fall back to the basic
        # translator than to wait while Ollama restarts or reloads the model.
        # Build the Gemma turns ourselves and send them raw. Through Ollama's chat template the
        # model starts a hidden "thinking" step on longer or formal text, uses up the token limit
        # there, and returns an empty answer.
        turns = [("system", _system_prompt(source, target))]
        for src, dst in EXAMPLES.get((source, target), []):
            turns += [("user", src), ("model", dst)]
        turns.append(("user", text))
        prompt = "".join(f"<|turn>{role}\n{content}<turn|>\n" for role, content in turns) + "<|turn>model\n"
        body = _post("/api/generate", {
            "model": MODEL, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "60m",
            "options": {"temperature": 0.2, "num_ctx": 4096, "stop": ["<turn|>", "<|turn>"],
                        "num_predict": min(600, 60 + 4 * len(text.split()))},
        }, timeout=timeout)
        out = " ".join(json.loads(body)["response"].split())  # one line, single spaces
        out = re.sub(r"^(translation|terjemahan|traducción)\s*:\s*", "", out, flags=re.I)
        out = out.strip().strip('"“”').strip()
        if not out or LEAKED_REASONING.search(out):
            raise RuntimeError("the local AI gave no usable translation")
        return out
