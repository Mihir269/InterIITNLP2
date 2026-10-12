"""
Baseline wrapper: IndicF5 zero-shot voice cloning with pinned settings, 16 kHz output.

    from baseline.tts import synthesize, save_wav
    audio = synthesize(text, "data/prompts/V1.wav", prompt_transcript, seed=0)   # float32, 16 kHz
    save_wav("out.wav", audio)

Settings come from baseline/config.yaml. Any decoding value can be overridden per call, for
example synthesize(..., nfe_step=16, speed=0.9). Overrides are allowed in Tasks 3 and 4; the
baseline always uses the file's values.
"""
import inspect
import random
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_SR = 16000
_ENGINE = None


def load_config(path=None):
    import yaml
    with open(path or HERE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_float(a) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a).squeeze()
    if a.dtype == np.int16:
        return a.astype(np.float32) / 32768.0
    return a.astype(np.float32)


def resample(y: np.ndarray, sr: int, target: int = OUT_SR) -> np.ndarray:
    if sr == target:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=target).astype(np.float32)


def _quiet(*args, **kwargs):
    pass


def _filter_kwargs(fn, kwargs):
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), set()
    keep = {k: v for k, v in kwargs.items() if k in params}
    return keep, set(kwargs) - set(keep)


class Engine:
    """Loads IndicF5 once and synthesizes with explicit decoding settings."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        if self.cfg.get("length_fix", True):
            from . import patch_length
            patch_length.ensure_patched()
        import torch
        from transformers import AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            warnings.warn("No GPU found: synthesis will be very slow.")
        self.hf = AutoModel.from_pretrained(self.cfg["repo"], revision=self.cfg.get("revision", "main"),
                                            trust_remote_code=True)
        if self.cfg.get("length_fix", True):
            from . import patch_length
            if patch_length.patch_hf_modules():
                raise RuntimeError("The model's own code needed the length fix and has now been patched. "
                                   "Restart the Python session and run again (this happens once).")
        try:
            self.hf = self.hf.to(self.device)
        except Exception:
            pass

        from f5_tts.infer import utils_infer as U
        self.U = U
        self.ema = getattr(self.hf, "ema_model", None)
        self.vocoder = getattr(self.hf, "vocoder", None)
        self.direct = self.ema is not None and self.vocoder is not None
        if not self.direct:
            warnings.warn("Could not find ema_model/vocoder on the IndicF5 object; falling back to its "
                          "__call__, which ignores decoding overrides.")
        self._warned = set()

    def __call__(self, text, prompt_wav, prompt_text, seed=0, **overrides):
        settings = {**self.cfg["decoding"], **overrides}
        set_seed(seed)
        if self.direct:
            pre_kw, _ = _filter_kwargs(self.U.preprocess_ref_audio_text, {"show_info": _quiet})
            ref_audio, ref_text = self.U.preprocess_ref_audio_text(str(prompt_wav), prompt_text, **pre_kw)
            extra = {"show_info": _quiet, "device": self.device}
            kw, unknown = _filter_kwargs(self.U.infer_process, {**settings, **extra})
            unknown -= set(extra)
            if unknown - self._warned:
                warnings.warn(f"infer_process does not accept {sorted(unknown)}; ignored.")
                self._warned |= unknown
            wave, sr, _ = self.U.infer_process(ref_audio, ref_text, text, self.ema, self.vocoder, **kw)
        else:
            if overrides and "fallback" not in self._warned:
                warnings.warn("Decoding overrides are ignored in fallback mode.")
                self._warned.add("fallback")
            wave = self.hf(text, ref_audio_path=str(prompt_wav), ref_text=prompt_text)
            sr = self.cfg.get("model_sample_rate", 24000)
        return resample(to_float(wave), int(sr))


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


def synthesize(text: str, prompt_wav, prompt_text: str, seed: int = 0, **overrides) -> np.ndarray:
    """Clone the voice in `prompt_wav` (whose transcript is `prompt_text`) reading `text`.

    Returns mono float32 audio at 16 kHz. Keyword overrides change decoding settings
    (nfe_step, cfg_strength, sway_sampling_coef, speed, cross_fade_duration, target_rms).
    """
    return get_engine()(text, prompt_wav, prompt_text, seed=seed, **overrides)


def save_wav(path, audio: np.ndarray, sr: int = OUT_SR):
    """Write mono 16-bit PCM wav (the required submission format)."""
    import soundfile as sf
    y = np.clip(to_float(audio), -1.0, 1.0)
    if sr != OUT_SR:
        y, sr = resample(y, sr), OUT_SR
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
