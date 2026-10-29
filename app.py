"""Live Translator

- Listens to the other person (Spanish or Malay) on your speakers and shows English text.
- You speak English (hold F8 or the big button); they hear their language through the
  virtual mic (VB-Audio "CABLE Input" -> pick "CABLE Output" as mic in Zoom/Meet).
"""
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import ttk

import audio_io
import stt
import tts
from llm_translate import LLMTranslator
from translate import Translator

LANGUAGES = {"Spanish": "es", "Malay": "ms"}  # the other person's language
ENGINES = {"Natural (local AI)": "llm", "Basic (offline NLLB)": "nllb"}
MY_LANG = "en"
HOTKEY = "f8"
MODELS = ["large-v3-turbo", "medium", "small", "base"]
NONE = "(none)"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")

C = dict(bg="#1d1f24", panel="#2a2d34", hover="#353943", text="#e9e9ec", muted="#8f939c",
         src="#cfd2d9", them="#7cc4ff", me="#9be58f", err="#ff7a7a", ptt="#2f6f3e", ptt_on="#c0392b")


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


class App:
    def __init__(self, root):
        self.root = root
        self.settings = load_settings()
        self.ui_q = queue.Queue()
        self.listen_q = queue.Queue()
        self.speak_q = queue.Queue()
        self.translator = None  # NLLB: always loaded, also the fallback
        self.llm = None  # local AI (Ollama), loaded in the background
        self.llm_recovering = threading.Event()  # set while Ollama restarts; basic translator meanwhile
        self.transcriber = None
        self.model_gen = 0
        self.listener = None
        self.mic = None
        self.ptt_active = False
        self.playing = threading.Event()
        self.cfg = {}  # plain copy of the settings that worker threads read
        self.loops, self.mics, self.outs = {}, {}, {}

        self._build_ui()
        self._refresh_devices()
        threading.Thread(target=self._listen_worker, daemon=True).start()
        threading.Thread(target=self._speak_worker, daemon=True).start()
        self._load_model()
        self._setup_hotkey()
        root.after(50, self._poll_ui)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI ----------------

    def _build_ui(self):
        r = self.root
        r.title("Live Translator")
        r.geometry("860x760")
        r.minsize(660, 560)
        r.configure(bg=C["bg"])

        s = ttk.Style(r)
        s.theme_use("clam")
        s.configure(".", background=C["bg"], foreground=C["text"], fieldbackground=C["panel"],
                    bordercolor=C["panel"], font=("Segoe UI", 10))
        s.configure("Muted.TLabel", foreground=C["muted"])
        s.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))
        s.configure("TButton", background=C["panel"], padding=(10, 6))
        s.map("TButton", background=[("active", C["hover"]), ("disabled", C["bg"])],
              foreground=[("disabled", C["muted"])])
        s.configure("TCombobox", background=C["panel"], arrowcolor=C["text"])
        s.map("TCombobox", fieldbackground=[("readonly", C["panel"])],
              foreground=[("readonly", C["text"])], selectbackground=[("readonly", C["panel"])])
        s.configure("TCheckbutton", background=C["bg"])
        s.map("TCheckbutton", background=[("active", C["bg"])])
        r.option_add("*TCombobox*Listbox.background", C["panel"])
        r.option_add("*TCombobox*Listbox.foreground", C["text"])
        r.option_add("*TCombobox*Listbox.selectBackground", C["hover"])

        top = ttk.Frame(r, padding=(14, 12, 14, 6))
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        self.loop_var = tk.StringVar()
        self.mic_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.mon_var = tk.StringVar()
        self.mon_on = tk.BooleanVar(value=self.settings.get("monitor_on", True))
        lang = self.settings.get("language")
        self.lang_var = tk.StringVar(value=lang if lang in LANGUAGES else next(iter(LANGUAGES)))
        self.voice_var = tk.StringVar()
        engine = self.settings.get("engine")
        self.engine_var = tk.StringVar(value=engine if engine in ENGINES else next(iter(ENGINES)))
        self.model_var = tk.StringVar(value=self.settings.get("model", MODELS[0]))

        def row(y, label, var, col=0, width=40):
            ttk.Label(top, text=label).grid(row=y, column=col, sticky="w", padx=(0, 8), pady=4)
            cb = ttk.Combobox(top, textvariable=var, state="readonly", width=width)
            cb.grid(row=y, column=col + 1, sticky="ew", pady=4, padx=(0, 14))
            cb.bind("<<ComboboxSelected>>", self._on_setting_changed)
            return cb

        self.loop_cb = row(0, "Listen to (their voice):", self.loop_var)
        self.mic_cb = row(1, "My microphone:", self.mic_var)
        self.out_cb = row(2, "Send translated voice to:", self.out_var)
        ttk.Label(top, text="Choose “CABLE Input” above, then pick “CABLE Output” as the microphone "
                            "in WhatsApp / Zoom / Meet.",
                  style="Muted.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))

        ttk.Checkbutton(top, text="Also let me hear it on:", variable=self.mon_on,
                        command=self._on_setting_changed).grid(row=3, column=0, sticky="w", pady=4)
        self.mon_cb = ttk.Combobox(top, textvariable=self.mon_var, state="readonly", width=40)
        self.mon_cb.grid(row=3, column=1, sticky="ew", pady=4, padx=(0, 14))
        self.mon_cb.bind("<<ComboboxSelected>>", self._on_setting_changed)

        lang_cb = row(0, "Their language:", self.lang_var, col=2, width=26)
        lang_cb["values"] = list(LANGUAGES)
        lang_cb.bind("<<ComboboxSelected>>", lambda e: self._on_language_changed())
        self.voice_cb = row(1, "Voice:", self.voice_var, col=2, width=26)
        model_cb = row(2, "Speech model:", self.model_var, col=2, width=26)
        model_cb["values"] = MODELS
        model_cb.bind("<<ComboboxSelected>>", lambda e: (self._on_setting_changed(), self._load_model()))
        engine_cb = row(3, "Translation:", self.engine_var, col=2, width=26)
        engine_cb["values"] = list(ENGINES)

        btns = ttk.Frame(top)
        btns.grid(row=4, column=2, columnspan=2, sticky="e", pady=(4, 0))
        ttk.Button(btns, text="Clear", command=self._clear_log).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Refresh devices", command=self._refresh_devices).pack(side="left", padx=(0, 8))
        self.listen_btn = ttk.Button(btns, text="▶  Start listening", command=self._toggle_listen,
                                     state="disabled")
        self.listen_btn.pack(side="left")

        # conversation log
        mid = ttk.Frame(r, padding=(14, 6))
        mid.pack(fill="both", expand=True)
        self.log = tk.Text(mid, wrap="word", bg=C["panel"], fg=C["text"], bd=0, padx=14, pady=10,
                           insertbackground=C["text"], state="disabled", highlightthickness=0)
        sb = ttk.Scrollbar(mid, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.log.tag_configure("them_src", foreground=C["src"], font=("Segoe UI", 13), spacing1=14)
        self.log.tag_configure("them_dst", foreground=C["them"], font=("Segoe UI", 15, "bold"))
        self.log.tag_configure("me_src", foreground=C["src"], font=("Segoe UI", 11), spacing1=14)
        self.log.tag_configure("me_dst", foreground=C["me"], font=("Segoe UI", 12, "bold"))
        self.log.tag_configure("info", foreground=C["muted"], font=("Segoe UI", 9, "italic"), spacing1=6)
        self.log.tag_configure("error", foreground=C["err"], font=("Segoe UI", 9), spacing1=6)

        # talk controls
        bottom = ttk.Frame(r, padding=(14, 6, 14, 12))
        bottom.pack(fill="x")
        self.ptt_btn = tk.Button(bottom, text=f"🎤  Hold to talk  ({HOTKEY.upper()})", bg=C["ptt"],
                                 fg="white", activebackground=C["ptt_on"], activeforeground="white",
                                 font=("Segoe UI", 13, "bold"), bd=0, relief="flat", height=2,
                                 cursor="hand2")
        self.ptt_btn.pack(fill="x")
        self.ptt_btn.bind("<ButtonPress-1>", lambda e: self._ptt_down())
        self.ptt_btn.bind("<ButtonRelease-1>", lambda e: self._ptt_up())

        type_row = ttk.Frame(bottom)
        type_row.pack(fill="x", pady=(8, 0))
        self.entry = tk.Entry(type_row, bg=C["panel"], fg=C["text"], insertbackground=C["text"],
                              bd=0, font=("Segoe UI", 11), highlightthickness=1,
                              highlightbackground=C["panel"], highlightcolor=C["them"])
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", lambda e: self._send_typed())
        self.say_btn = ttk.Button(type_row, command=self._send_typed)
        self.say_btn.pack(side="left", padx=(8, 0))

        self.status = ttk.Label(bottom, text="Starting…", style="Muted.TLabel")
        self.status.pack(fill="x", pady=(8, 0))

        self._on_language_changed(save=False)
        self._write("info", "Type English below, or hold F8 / the green button and speak. "
                            "Press “Start listening” to see their words in English.")

    def _on_language_changed(self, save=True):
        lang = self.lang_var.get()
        voices = list(tts.VOICES[LANGUAGES[lang]])
        self.voice_cb["values"] = voices
        saved = self.settings.get(f"voice_{LANGUAGES[lang]}")
        self.voice_var.set(saved if saved in voices else voices[0])
        self.say_btn.configure(text=f"Say in {lang}")
        if save:
            self._on_setting_changed()

    def _write(self, tag, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.configure(state="disabled")
        self.log.see("end")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_status(self, text):
        self.status.configure(text=text)

    # ---------------- devices & settings ----------------

    def _refresh_devices(self):
        try:
            self.loops, loop_default = audio_io.list_loopbacks()
            self.mics, mic_default = audio_io.list_inputs()
            self.outs, out_default = audio_io.list_outputs()
        except Exception as e:
            self._write("error", f"Could not read audio devices: {e}")
            return
        cable = next((n for n in self.outs if "CABLE Input" in n), None)
        if cable is None:
            self._write("error", "VB-Audio Virtual Cable not found. Install it from vb-audio.com/Cable.")

        def pick(cb, var, names, saved, default):
            cb["values"] = names
            var.set(saved if saved in names else default if default in names else (names[0] if names else ""))

        pick(self.loop_cb, self.loop_var, list(self.loops), self.settings.get("loopback"), loop_default)
        pick(self.mic_cb, self.mic_var, list(self.mics), self.settings.get("mic"), mic_default)
        pick(self.out_cb, self.out_var, list(self.outs), self.settings.get("output"), cable)
        speakers = [n for n in self.outs if "CABLE" not in n]
        pick(self.mon_cb, self.mon_var, speakers, self.settings.get("monitor"), out_default)
        self._on_setting_changed()

    def _on_setting_changed(self, *_):
        old_mic = self.cfg.get("mic")
        lang = LANGUAGES[self.lang_var.get()]
        voices = tts.VOICES[lang]
        self.cfg = {
            "loopback": self.loops.get(self.loop_var.get()),
            "mic": self.mics.get(self.mic_var.get()),
            "output": self.outs.get(self.out_var.get()),
            "monitor": self.outs.get(self.mon_var.get()) if self.mon_on.get() else None,
            "their_lang": lang,
            "engine": ENGINES[self.engine_var.get()],
            "voice": voices.get(self.voice_var.get(), next(iter(voices.values()))),
        }
        self.settings.update(loopback=self.loop_var.get(), mic=self.mic_var.get(),
                             output=self.out_var.get(), monitor=self.mon_var.get(),
                             monitor_on=self.mon_on.get(), language=self.lang_var.get(),
                             engine=self.engine_var.get(),
                             model=self.model_var.get())
        self.settings[f"voice_{lang}"] = self.voice_var.get()
        if self.cfg["mic"] != old_mic and self.transcriber is not None:
            self._open_mic()

    def _save_settings(self):
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass

    # ---------------- model ----------------

    def _load_model(self):
        self.model_gen += 1
        gen, name = self.model_gen, self.model_var.get()
        self.transcriber = None
        self.listen_btn.configure(state="disabled")
        self._set_status(f"Loading models (speech: “{name}”)… The first time, they download (about 2 GB).")

        def run():
            log = lambda m: self.ui_q.put(("info", m))  # noqa: E731
            try:
                if self.translator is None:
                    print("loading translator", flush=True)
                    self.translator = Translator(log=log)
                print(f"loading whisper {name}", flush=True)
                t = stt.Transcriber(name, log=log)
                print("models ready", flush=True)
                if self.llm is None:
                    threading.Thread(target=self._load_llm, daemon=True).start()
            except Exception as e:
                traceback.print_exc()
                self.ui_q.put(("error", f"Could not load speech model: {e}"))
                return
            if gen == self.model_gen:
                self.ui_q.put(("model_ready", t))

        threading.Thread(target=run, daemon=True).start()

    def _load_llm(self):
        log = lambda m: self.ui_q.put(("info", m))  # noqa: E731
        try:
            print("loading local AI translator", flush=True)
            self.llm = LLMTranslator(log=log)
            print("local AI translator ready", flush=True)
            log("Natural translation (local AI) is ready.")
        except Exception as e:
            traceback.print_exc()
            log(f"Natural translation not available ({e}). Using the basic translator.")

    def _translate(self, text, source, target):
        if self.cfg.get("engine") == "llm" and self.llm is not None and not self.llm_recovering.is_set():
            try:
                return self.llm.translate(text, source, target)
            except Exception as e:
                traceback.print_exc()
                self.ui_q.put(("info", f"Local AI translator failed ({e}). Used the basic one."))
                if isinstance(e, OSError):  # Ollama crashed or stopped: bring it back in the background
                    self.llm_recovering.set()
                    threading.Thread(target=self._recover_llm, daemon=True).start()
        return self.translator.translate(text, source, target)

    def _recover_llm(self):
        try:
            for _ in range(3):
                try:
                    self.llm.warm_up()
                    self.ui_q.put(("info", "Local AI translator is back."))
                    return
                except Exception:
                    traceback.print_exc()
                    time.sleep(3)
            self.ui_q.put(("info", "Local AI translator did not come back. Using the basic one."))
        finally:
            self.llm_recovering.clear()

    def _open_mic(self):
        if self.mic:
            try:
                self.mic.close()
            except Exception:
                pass
            self.mic = None
        if self.cfg.get("mic") is None:
            return
        try:
            mic = audio_io.MicRecorder(self.cfg["mic"])
            mic.open()
            self.mic = mic
        except Exception as e:
            self._write("error", f"Could not open microphone: {e}")

    # ---------------- listening (them -> me) ----------------

    def _toggle_listen(self):
        if self.listener:
            self.listener.stop()
            self.listener = None
            self.listen_btn.configure(text="▶  Start listening")
            self._set_status("Stopped listening.")
            return
        if self.cfg.get("loopback") is None:
            self._write("error", "Choose a device in “Listen to”.")
            return
        try:
            # Mute while our translated voice plays on the speakers, so we don't translate ourselves.
            self.listener = audio_io.LoopbackListener(
                self.cfg["loopback"], on_segment=self.listen_q.put,
                is_muted=lambda: self.playing.is_set() and self.cfg.get("monitor") is not None)
            self.listener.start()
        except Exception as e:
            self.listener = None
            self._write("error", f"Could not start listening: {e}")
            return
        self.listen_btn.configure(text="■  Stop listening")
        self._set_status(f"Listening to: {self.loop_var.get()}")

    def _listen_worker(self):
        while True:
            audio = self.listen_q.get()
            t = self.transcriber
            if t is None:
                continue
            lang = self.cfg["their_lang"]
            try:
                src = t.transcribe(audio, lang)
                if not src:
                    continue
                self.ui_q.put(("them", lang, src, self._translate(src, lang, MY_LANG)))
            except Exception as e:
                self.ui_q.put(("error", f"Listening error: {e}"))

    # ---------------- speaking (me -> them) ----------------

    def _ptt_down(self):
        if self.ptt_active:
            return  # key auto-repeat
        if self.transcriber is None or self.mic is None:
            self.ui_q.put(("status", "Wait, the speech model or microphone is not ready yet."))
            return
        self.ptt_active = True
        self.mic.begin()
        self.ui_q.put(("ptt", True))

    def _ptt_up(self):
        if not self.ptt_active:
            return
        self.ptt_active = False
        audio = self.mic.end()
        self.ui_q.put(("ptt", False))
        if len(audio) < audio_io.TARGET_SR * 0.4:
            self.ui_q.put(("status", "Too short. Hold the key while you speak."))
            return
        audio, peak = audio_io.clean_voice(audio)
        self._save_last_recording(audio)
        if peak < 0.005:
            self.ui_q.put(("error", f"The mic heard almost nothing (level {peak * 100:.1f}%). Check that "
                                    "“My microphone” is the mic you speak into, that it is not muted, "
                                    "and its volume in Windows Sound settings."))
            self.ui_q.put(("status", "Ready."))
            return
        self.speak_q.put(("audio", (audio, peak)))

    def _save_last_recording(self, audio):
        """Keep the last push-to-talk recording as last_recording.wav, to check what the mic heard."""
        try:
            import wave
            with wave.open(os.path.join(APP_DIR, "last_recording.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(audio_io.TARGET_SR)
                w.writeframes((audio.clip(-1, 1) * 32767).astype("<i2").tobytes())
        except Exception:
            traceback.print_exc()

    def _send_typed(self):
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self.speak_q.put(("text", text))

    def _speak_worker(self):
        while True:
            kind, payload = self.speak_q.get()
            try:
                if kind == "audio":
                    audio, peak = payload
                    self.ui_q.put(("status", "Understanding your English…"))
                    src = self.transcriber.transcribe(audio, MY_LANG, vad=False)
                    if not src:
                        tip = " Your mic is quiet: speak closer or raise its volume." if peak < 0.05 else ""
                        self.ui_q.put(("info", f"I didn't catch that (mic level {peak * 100:.0f}%).{tip} "
                                               "Please try again."))
                        self.ui_q.put(("status", "Ready."))
                        continue
                else:
                    src = payload
                lang, voice = self.cfg["their_lang"], self.cfg["voice"]
                dst = self._translate(src, MY_LANG, lang)
                self.ui_q.put(("me", lang, src, dst))
                if kind == "text" and dst.strip():
                    self.ui_q.put(("copy", dst))  # typed text: ready to paste with Ctrl+V
                if not dst.strip():
                    self.ui_q.put(("error", "The translation came back empty, so nothing was spoken. Try again."))
                    self.ui_q.put(("status", "Ready."))
                    continue
                self.ui_q.put(("status", "Speaking…"))
                try:
                    audio, sr = tts.synthesize(dst, voice)
                except Exception as e:
                    traceback.print_exc()
                    self.ui_q.put(("error", "Voice not played: Microsoft's online voice service did not answer "
                                            f"after 3 tries ({type(e).__name__}). Check internet / VPN, "
                                            "then type it again."))
                    self.ui_q.put(("status", "Ready."))
                    continue
                self.playing.set()
                try:
                    audio_io.play(audio, sr, [self.cfg.get("output"), self.cfg.get("monitor")])
                finally:
                    time.sleep(0.3)
                    self.playing.clear()
                self.ui_q.put(("status", "Ready."))
            except Exception as e:
                self.ui_q.put(("error", f"Speaking error: {e}"))
                self.ui_q.put(("status", "Ready."))

    def _setup_hotkey(self):
        try:
            import keyboard
            keyboard.on_press_key(HOTKEY, lambda e: self._ptt_down())
            keyboard.on_release_key(HOTKEY, lambda e: self._ptt_up())
        except Exception as e:
            self._write("error", f"Global {HOTKEY.upper()} key not available ({e}). Use the button instead.")

    # ---------------- UI message loop ----------------

    def _poll_ui(self):
        try:
            while True:
                msg = self.ui_q.get_nowait()
                kind = msg[0]
                if kind == "them":
                    _, lang, src, dst = msg
                    self._write("them_src", f"Them ({lang.upper()}):  {src}")
                    self._write("them_dst", "EN:  " + dst)
                elif kind == "me":
                    _, lang, src, dst = msg
                    self._write("me_src", "You (EN):  " + src)
                    self._write("me_dst", f"→ {lang.upper()}:  {dst}")
                elif kind == "info":
                    self._write("info", msg[1])
                elif kind == "error":
                    self._write("error", msg[1])
                elif kind == "status":
                    self._set_status(msg[1])
                elif kind == "copy":
                    self.root.clipboard_clear()
                    self.root.clipboard_append(msg[1])
                    self.root.update()  # keeps the clipboard text after the window loses focus
                    self._write("info", "Copied to clipboard. Paste with Ctrl+V.")
                elif kind == "ptt":
                    on = msg[1]
                    self.ptt_btn.configure(bg=C["ptt_on"] if on else C["ptt"],
                                           text="🔴  Recording… release to translate" if on
                                           else f"🎤  Hold to talk  ({HOTKEY.upper()})")
                    if on:
                        self._set_status("Recording…")
                elif kind == "model_ready":
                    self.transcriber = msg[1]
                    self._open_mic()
                    self.listen_btn.configure(state="normal")
                    self._set_status(f"Ready. Speech model “{self.model_var.get()}” "
                                     f"running on {self.transcriber.device.upper()}.")
        except queue.Empty:
            pass
        if self.ptt_active and self.mic:
            # Live mic level while recording: full bar at 30% loudness, which is normal speech.
            filled = min(20, int(self.mic.level / 0.3 * 20))
            self._set_status(f"Recording…   mic level  {'█' * filled}{'░' * (20 - filled)}  "
                             f"{self.mic.level * 100:.0f}%")
        self.root.after(50, self._poll_ui)

    def _on_close(self):
        self._save_settings()
        try:
            import keyboard
            keyboard.unhook_all()
        except Exception:
            pass
        for thing in (self.listener, self.mic):
            try:
                if thing is self.listener and thing:
                    thing.stop()
                elif thing:
                    thing.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    # pythonw (and a future .exe) has no console: send prints and errors to a log file.
    log_file = open(os.path.join(APP_DIR, "app.log"), "w", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log_file
    if sys.stderr is None:
        sys.stderr = log_file
    threading.excepthook = lambda a: traceback.print_exception(a.exc_type, a.exc_value, a.exc_traceback)
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sharp text on high-DPI screens
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
