"""Audio devices, speaker capture (WASAPI loopback), push-to-talk mic, and playback."""
import collections
import queue
import threading

import numpy as np
import pyaudiowpatch as pyaudio
import sounddevice as sd
import soxr

TARGET_SR = 16000  # what Whisper expects


# ---------- device lists ----------

def _wasapi():
    for i, api in enumerate(sd.query_hostapis()):
        if "WASAPI" in api["name"]:
            return i, api
    raise RuntimeError("WASAPI audio is not available on this computer")


def list_inputs():
    """Microphones -> ({name: index}, default_name)."""
    api_idx, api = _wasapi()
    devs = {d["name"]: d["index"] for d in sd.query_devices()
            if d["hostapi"] == api_idx and d["max_input_channels"] > 0}
    default = sd.query_devices(api["default_input_device"])["name"] if api["default_input_device"] >= 0 else None
    return devs, default


def list_outputs():
    """Speakers / virtual cables -> ({name: index}, default_name)."""
    api_idx, api = _wasapi()
    devs = {d["name"]: d["index"] for d in sd.query_devices()
            if d["hostapi"] == api_idx and d["max_output_channels"] > 0}
    default = sd.query_devices(api["default_output_device"])["name"] if api["default_output_device"] >= 0 else None
    return devs, default


def list_loopbacks():
    """Speakers we can record from -> ({name: pyaudio_index}, default_name)."""
    p = pyaudio.PyAudio()
    try:
        devs = {d["name"]: d["index"] for d in p.get_loopback_device_info_generator()}
        try:
            default = p.get_default_wasapi_loopback()["name"]
        except Exception:
            default = None
        return devs, default
    finally:
        p.terminate()


# ---------- speech segmentation ----------

class SpeechSegmenter:
    """Cuts a continuous 16 kHz stream into sentences using loudness and pauses."""

    FRAME = 480  # 30 ms

    def __init__(self, on_segment, silence_sec=0.7, max_sec=12.0, min_speech_sec=0.4):
        self.on_segment = on_segment
        self.silence_frames = int(silence_sec * TARGET_SR / self.FRAME)
        self.max_frames = int(max_sec * TARGET_SR / self.FRAME)
        self.min_voiced = int(min_speech_sec * TARGET_SR / self.FRAME)
        self.noise = 0.003
        self.pending = np.zeros(0, np.float32)
        self.preroll = collections.deque(maxlen=10)
        self.reset()

    def reset(self):
        self.active = False
        self.chunks = []
        self.voiced = 0
        self.silent = 0
        self.preroll.clear()

    def feed(self, x):
        self.pending = np.concatenate([self.pending, x])
        while len(self.pending) >= self.FRAME:
            frame, self.pending = self.pending[:self.FRAME], self.pending[self.FRAME:]
            self._frame(frame)

    def _frame(self, f):
        rms = float(np.sqrt(np.mean(f * f))) + 1e-9
        loud = rms > max(0.006, self.noise * 3.0)
        if not self.active:
            if not loud:
                self.noise = 0.95 * self.noise + 0.05 * rms
            self.preroll.append(f)
            if loud:
                self.active = True
                self.chunks = list(self.preroll)
                self.voiced, self.silent = 1, 0
            return
        self.chunks.append(f)
        if loud:
            self.voiced += 1
            self.silent = 0
        else:
            self.silent += 1
        if self.silent >= self.silence_frames or len(self.chunks) >= self.max_frames:
            self.flush()

    def flush(self):
        """Send what we have (also called when the speaker goes fully quiet)."""
        if self.active and self.voiced >= self.min_voiced:
            self.on_segment(np.concatenate(self.chunks))
        self.reset()


# ---------- their voice: record what plays on the speakers ----------

class LoopbackListener:
    def __init__(self, device_index, on_segment, is_muted=lambda: False):
        self.device_index = device_index
        self.is_muted = is_muted
        self.segmenter = SpeechSegmenter(on_segment)
        self.q = queue.Queue()
        self.running = False

    def start(self):
        self.p = pyaudio.PyAudio()
        info = self.p.get_device_info_by_index(self.device_index)
        self.channels = int(info["maxInputChannels"])
        rate = int(info["defaultSampleRate"])
        self.resampler = soxr.ResampleStream(rate, TARGET_SR, 1, dtype="float32")
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=self.channels, rate=rate,
                                  input=True, input_device_index=self.device_index,
                                  frames_per_buffer=1024, stream_callback=self._callback)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _callback(self, data, frames, time_info, status):
        self.q.put(data)
        return None, pyaudio.paContinue

    def _run(self):
        while self.running:
            try:
                data = self.q.get(timeout=0.3)
            except queue.Empty:
                # Windows sends no data when nothing is playing: treat as end of sentence.
                self.segmenter.flush()
                continue
            if self.is_muted():
                self.segmenter.reset()  # don't translate our own Spanish voice
                continue
            x = np.frombuffer(data, np.float32).reshape(-1, self.channels).mean(axis=1)
            self.segmenter.feed(self.resampler.resample_chunk(x))

    def stop(self):
        self.running = False
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.p.terminate()


# ---------- my voice: push-to-talk microphone ----------

class MicRecorder:
    """Keeps the mic open so the first word is not cut; only stores audio while recording."""

    def __init__(self, device_index):
        dev = sd.query_devices(device_index)
        self.rate = int(dev["default_samplerate"])
        self.channels = int(dev["max_input_channels"])
        self.device_index = device_index
        self.lock = threading.Lock()
        self.recording = False
        self.chunks = []
        self.preroll = collections.deque(maxlen=8)
        self.level = 0.0  # loudness of the last 50 ms, 0..1, for the level bar

    def open(self):
        self.stream = sd.InputStream(device=self.device_index, samplerate=self.rate,
                                     channels=self.channels, dtype="float32",
                                     blocksize=int(self.rate * 0.05), callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        mono = indata.mean(axis=1).copy()
        self.level = float(np.max(np.abs(mono - mono.mean())))  # ignore the mic's DC offset
        with self.lock:
            if self.recording:
                self.chunks.append(mono)
            else:
                self.preroll.append(mono)

    def begin(self):
        with self.lock:
            self.chunks = list(self.preroll)[-4:]  # keep ~0.2 s from just before the key press
            self.recording = True

    def end(self):
        with self.lock:
            self.recording = False
            chunks, self.chunks = self.chunks, []
        if not chunks:
            return np.zeros(0, np.float32)
        return soxr.resample(np.concatenate(chunks), self.rate, TARGET_SR).astype(np.float32)


def clean_voice(audio):
    """Remove the mic's DC offset and boost quiet recordings. Returns (audio, peak before boost)."""
    if len(audio) == 0:
        return audio, 0.0
    audio = audio - audio.mean()
    peak = float(np.max(np.abs(audio)))
    if 0 < peak < 0.5:
        audio = audio * min(0.5 / peak, 30.0)  # at most 30x, so pure silence isn't blown up to noise
    return audio.astype(np.float32), peak

    def close(self):
        self.stream.stop()
        self.stream.close()


# ---------- playback ----------

def _play_on(device_index, audio, sr):
    dev = sd.query_devices(device_index)
    rate = int(dev["default_samplerate"])
    channels = min(int(dev["max_output_channels"]), 2)
    y = soxr.resample(audio, sr, rate).astype(np.float32)
    y = np.repeat(y[:, None], channels, axis=1)
    pad = np.zeros((int(rate * 0.15), channels), np.float32)  # avoid clipping the last word
    with sd.OutputStream(device=device_index, samplerate=rate, channels=channels, dtype="float32") as s:
        s.write(np.concatenate([y, pad]))


def play(audio, sr, device_indexes):
    """Play the same sound on several devices at once and wait until done."""
    threads = [threading.Thread(target=_play_on, args=(i, audio, sr), daemon=True)
               for i in device_indexes if i is not None]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
