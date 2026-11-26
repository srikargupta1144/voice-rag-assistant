"""
File: tests/unit/test_voice_wrappers.py

Description:
    Unit tests for voice/stt.py (WhisperSTT, get_stt), voice/tts.py (PiperTTS, get_tts) and
    the WAV helpers in voice/audio.py. The faster_whisper and piper packages are replaced by
    fake modules, so no model is loaded and no file is downloaded.

Libraries used:
    io - standard library; reads the audio buffer given to the fake model
    sys - standard library; places the fake faster_whisper and piper modules in the cache
    types - standard library; builds the fake modules and simple objects
    wave - standard library; checks the WAV header of the synthesized audio
    pytest - third party; fixtures, parametrisation and monkeypatching
    voice.audio - project; WAV helpers under test
    voice.stt - project; WhisperSTT, Transcript and get_stt under test
    voice.tts - project; PiperTTS and get_tts under test

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

import io
import sys
import types
import wave

import pytest

import voice.stt as stt_module
import voice.tts as tts_module
from voice.audio import concat_wav, pcm_to_wav, read_wav, wav_duration_s
from voice.stt import Transcript, WhisperSTT, get_stt
from voice.tts import PiperTTS, get_tts


class FakeWhisperModel:
    """
    Class name: FakeWhisperModel
    Purpose: Replacement for faster_whisper.WhisperModel. It records how it was created and
        called and returns scripted segments and audio information.
    """

    instances = []
    segments = [types.SimpleNamespace(text=" hello "), types.SimpleNamespace(text="world  ")]
    info = types.SimpleNamespace(duration=2.5, language="en")

    def __init__(self, model, device=None, compute_type=None, cpu_threads=None):
        """
        Function name: FakeWhisperModel.__init__
        Purpose: Record the constructor arguments and register the instance.
        Parameters:
            model (str): Model folder or size name.
            device (str | None): Device name.
            compute_type (str | None): Compute type.
            cpu_threads (int | None): Thread count.
        Returns:
            None: Initialises the instance.
        """
        self.init = {
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
        }
        self.calls = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        """
        Function name: FakeWhisperModel.transcribe
        Purpose: Record the audio bytes and the options, and return the scripted result.
        Parameters:
            audio (BytesIO): File-like audio buffer.
            kwargs (dict): Decoding options.
        Returns:
            tuple: (iterator of segments, info). The segments are a one-shot iterator, as in
                the real library.
        """
        self.calls.append((audio.read(), kwargs))
        return iter(type(self).segments), type(self).info


@pytest.fixture
def fake_whisper(monkeypatch):
    """
    Function name: fake_whisper
    Purpose: Fixture that installs a fake faster_whisper module and clears the process-wide
        speech-to-text instance before and after the test.
    Parameters:
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        type: The FakeWhisperModel class, with an empty instance list.
    """
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(stt_module, "_stt", None)
    monkeypatch.setattr(FakeWhisperModel, "instances", [])
    monkeypatch.setattr(
        FakeWhisperModel,
        "segments",
        [types.SimpleNamespace(text=" hello "), types.SimpleNamespace(text="world  ")],
    )
    return FakeWhisperModel


def test_whisper_stt_loads_model_on_cpu(fake_whisper):
    """
    Function name: test_whisper_stt_loads_model_on_cpu
    Purpose: Check that WhisperSTT creates the model on the CPU with the given model name,
        compute type and thread count.
    Parameters:
        fake_whisper (type): Fake model class fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    WhisperSTT("small", "float32", 4)
    assert fake_whisper.instances[0].init == {
        "model": "small",
        "device": "cpu",
        "compute_type": "float32",
        "cpu_threads": 4,
    }


def test_transcribe_joins_and_strips_segments(fake_whisper):
    """
    Function name: test_transcribe_joins_and_strips_segments
    Purpose: Check that the segment texts are stripped, joined with one space and returned
        with the duration as a float and the language from the audio information.
    Parameters:
        fake_whisper (type): Fake model class fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    result = WhisperSTT("small").transcribe(b"RIFFdata")
    assert result == Transcript(text="hello world", duration_s=2.5, language="en")
    assert isinstance(result.duration_s, float)


def test_transcribe_passes_decoding_options(fake_whisper):
    """
    Function name: test_transcribe_passes_decoding_options
    Purpose: Check that the audio bytes, language, beam size, VAD filter and initial prompt
        are passed to the model, with the fixed VAD parameters and no previous-text
        conditioning.
    Parameters:
        fake_whisper (type): Fake model class fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    stt = WhisperSTT("small", language="hi", beam_size=5, vad=False, initial_prompt="Kova Bank")
    stt.transcribe(b"audio-bytes")
    audio, kwargs = fake_whisper.instances[0].calls[0]
    assert audio == b"audio-bytes"
    assert kwargs == {
        "language": "hi",
        "beam_size": 5,
        "vad_filter": False,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "condition_on_previous_text": False,
        "initial_prompt": "Kova Bank",
    }


def test_transcribe_defaults(fake_whisper):
    """
    Function name: test_transcribe_defaults
    Purpose: Check the default options: English, greedy decoding, VAD on and an empty initial
        prompt turned into None.
    Parameters:
        fake_whisper (type): Fake model class fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    stt = WhisperSTT("small")
    assert (stt.language, stt.beam_size, stt.vad, stt.initial_prompt) == ("en", 1, True, None)
    stt.transcribe(b"x")
    kwargs = fake_whisper.instances[0].calls[0][1]
    assert kwargs["language"] == "en" and kwargs["beam_size"] == 1
    assert kwargs["vad_filter"] is True and kwargs["initial_prompt"] is None


def test_transcribe_with_no_segments_gives_empty_text(fake_whisper, monkeypatch):
    """
    Function name: test_transcribe_with_no_segments_gives_empty_text
    Purpose: Check that silence (no segments) gives an empty transcript text.
    Parameters:
        fake_whisper (type): Fake model class fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.setattr(FakeWhisperModel, "segments", [])
    assert WhisperSTT("small").transcribe(b"x").text == ""


def test_transcribe_ignores_blank_edge_segments(fake_whisper, monkeypatch):
    """
    Function name: test_transcribe_ignores_blank_edge_segments
    Purpose: Check that segments made only of spaces do not leave extra spaces at the ends of
        the text.
    Parameters:
        fake_whisper (type): Fake model class fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    segs = [types.SimpleNamespace(text="  "), types.SimpleNamespace(text="hi")]
    monkeypatch.setattr(FakeWhisperModel, "segments", segs)
    assert WhisperSTT("small").transcribe(b"x").text == "hi"


def test_get_stt_uses_settings_and_caches(fake_whisper, settings):
    """
    Function name: test_get_stt_uses_settings_and_caches
    Purpose: Check that get_stt builds one model from the stt settings, returns the same
        object on later calls even with other settings, and stores it in the module slot.
    Parameters:
        fake_whisper (type): Fake model class fixture.
        settings (Settings): Test settings fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    first = get_stt(settings)
    other = settings.model_copy(
        update={"stt": settings.stt.model_copy(update={"model_size": "tiny"})}
    )
    assert get_stt(other) is first and stt_module._stt is first
    assert len(fake_whisper.instances) == 1
    s = settings.stt
    assert (first.language, first.beam_size, first.vad) == (s.language, s.beam_size, s.vad)
    assert first.initial_prompt == s.initial_prompt
    assert fake_whisper.instances[0].init["compute_type"] == s.compute_type


def test_get_stt_downloads_by_size_when_no_local_model(fake_whisper, settings, tmp_path):
    """
    Function name: test_get_stt_downloads_by_size_when_no_local_model
    Purpose: Check that the model size name is used when the local folder has no model.bin.
    Parameters:
        fake_whisper (type): Fake model class fixture.
        settings (Settings): Test settings fixture.
        tmp_path (Path): pytest temporary directory.
    Returns:
        None: The test passes or fails through assertions.
    """
    empty = tmp_path / "whisper"
    empty.mkdir()
    s = settings.model_copy(
        update={
            "stt": settings.stt.model_copy(update={"model_dir": empty, "model_size": "base.en"})
        }
    )
    get_stt(s)
    assert fake_whisper.instances[0].init["model"] == "base.en"


def test_get_stt_uses_local_model_folder(fake_whisper, settings, tmp_path):
    """
    Function name: test_get_stt_uses_local_model_folder
    Purpose: Check that the local model folder is used when it contains model.bin.
    Parameters:
        fake_whisper (type): Fake model class fixture.
        settings (Settings): Test settings fixture.
        tmp_path (Path): pytest temporary directory.
    Returns:
        None: The test passes or fails through assertions.
    """
    local = tmp_path / "whisper"
    local.mkdir()
    (local / "model.bin").write_bytes(b"x")
    s = settings.model_copy(
        update={"stt": settings.stt.model_copy(update={"model_dir": local, "model_size": "tiny"})}
    )
    get_stt(s)
    assert fake_whisper.instances[0].init["model"] == str(local)


class FakeChunk:
    """
    Class name: FakeChunk
    Purpose: One piece of synthesized audio, as returned by the Piper voice.
    """

    def __init__(self, data: bytes):
        """
        Function name: FakeChunk.__init__
        Purpose: Store the PCM bytes.
        Parameters:
            data (bytes): 16-bit PCM audio.
        Returns:
            None: Initialises the instance.
        """
        self.audio_int16_bytes = data


class FakePiperVoice:
    """
    Class name: FakePiperVoice
    Purpose: Replacement for piper.PiperVoice. It records the loaded path and every
        synthesis call, and returns two scripted audio chunks.
    """

    loaded = []
    rate = 22050

    def __init__(self):
        """
        Function name: FakePiperVoice.__init__
        Purpose: Set the voice configuration (sample rate) and an empty call log.
        Parameters:
            None
        Returns:
            None: Initialises the instance.
        """
        self.config = types.SimpleNamespace(sample_rate=type(self).rate)
        self.calls = []
        self.chunks = [FakeChunk(b"\x01\x00" * 10), FakeChunk(b"\x02\x00" * 5)]

    @classmethod
    def load(cls, path):
        """
        Function name: FakePiperVoice.load
        Purpose: Record the path and return a new voice.
        Parameters:
            path (str): Path of the .onnx voice file.
        Returns:
            FakePiperVoice: The new voice.
        """
        voice = cls()
        cls.loaded.append((path, voice))
        return voice

    def synthesize(self, text, config):
        """
        Function name: FakePiperVoice.synthesize
        Purpose: Record the text and configuration and return the scripted chunks.
        Parameters:
            text (str): Text to speak.
            config (FakeSynthesisConfig): Synthesis options.
        Returns:
            list[FakeChunk]: The audio chunks (none for empty text).
        """
        self.calls.append((text, config))
        return self.chunks if text else []


class FakeSynthesisConfig:
    """
    Class name: FakeSynthesisConfig
    Purpose: Replacement for piper.SynthesisConfig that keeps the length scale.
    """

    def __init__(self, length_scale=1.0):
        """
        Function name: FakeSynthesisConfig.__init__
        Purpose: Store the length scale.
        Parameters:
            length_scale (float): Speaking speed factor.
        Returns:
            None: Initialises the instance.
        """
        self.length_scale = length_scale


@pytest.fixture
def fake_piper(monkeypatch):
    """
    Function name: fake_piper
    Purpose: Fixture that installs a fake piper module and clears the process-wide
        text-to-speech instance before and after the test.
    Parameters:
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        type: The FakePiperVoice class, with an empty load log and the default sample rate.
    """
    module = types.ModuleType("piper")
    module.PiperVoice = FakePiperVoice
    module.SynthesisConfig = FakeSynthesisConfig
    monkeypatch.setitem(sys.modules, "piper", module)
    monkeypatch.setattr(tts_module, "_tts", None)
    monkeypatch.setattr(FakePiperVoice, "loaded", [])
    monkeypatch.setattr(FakePiperVoice, "rate", 22050)
    return FakePiperVoice


def test_piper_tts_loads_voice_and_reads_sample_rate(fake_piper):
    """
    Function name: test_piper_tts_loads_voice_and_reads_sample_rate
    Purpose: Check that PiperTTS loads the given path, takes the sample rate from the voice
        configuration as an int and passes the length scale to the synthesis options.
    Parameters:
        fake_piper (type): Fake voice class fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    fake_piper.rate = 16000
    tts = PiperTTS("voice.onnx", length_scale=1.3)
    assert fake_piper.loaded[0][0] == "voice.onnx"
    assert tts.sample_rate == 16000 and isinstance(tts.sample_rate, int)
    assert tts._config.length_scale == 1.3


def test_synthesize_returns_valid_wav_with_header(fake_piper):
    """
    Function name: test_synthesize_returns_valid_wav_with_header
    Purpose: Check that the audio chunks are joined into one mono 16-bit WAV at the voice
        sample rate, and that the text and options reach the voice.
    Parameters:
        fake_piper (type): Fake voice class fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    tts = PiperTTS("voice.onnx", 0.9)
    wav = tts.synthesize("Hello there.")
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 22050)
        assert w.getnframes() == 15
        assert w.readframes(15) == b"\x01\x00" * 10 + b"\x02\x00" * 5
    text, config = fake_piper.loaded[0][1].calls[0]
    assert text == "Hello there." and config.length_scale == 0.9


def test_synthesize_empty_text_gives_empty_wav(fake_piper):
    """
    Function name: test_synthesize_empty_text_gives_empty_wav
    Purpose: Check that empty text gives a valid WAV file with no frames.
    Parameters:
        fake_piper (type): Fake voice class fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    wav = PiperTTS("voice.onnx").synthesize("")
    pcm, rate, channels, width = read_wav(wav)
    assert pcm == b"" and (rate, channels, width) == (22050, 1, 2)
    assert wav_duration_s(wav) == 0.0


def test_get_tts_loads_configured_voice_and_caches(fake_piper, settings, tmp_path):
    """
    Function name: test_get_tts_loads_configured_voice_and_caches
    Purpose: Check that get_tts loads <voice_dir>/<voice>.onnx once with the configured length
        scale and returns the same object on later calls.
    Parameters:
        fake_piper (type): Fake voice class fixture.
        settings (Settings): Test settings fixture.
        tmp_path (Path): pytest temporary directory.
    Returns:
        None: The test passes or fails through assertions.
    """
    voice_dir = tmp_path / "piper"
    voice_dir.mkdir()
    (voice_dir / "my-voice.onnx").write_bytes(b"x")
    s = settings.model_copy(
        update={
            "tts": settings.tts.model_copy(
                update={"voice": "my-voice", "voice_dir": voice_dir, "length_scale": 1.2}
            )
        }
    )
    first = get_tts(s)
    assert get_tts(s) is first and tts_module._tts is first
    assert len(fake_piper.loaded) == 1
    assert fake_piper.loaded[0][0] == str(voice_dir / "my-voice.onnx")
    assert first._config.length_scale == 1.2 and first.sample_rate == 22050


def test_get_tts_missing_voice_raises_with_download_hint(fake_piper, settings, tmp_path):
    """
    Function name: test_get_tts_missing_voice_raises_with_download_hint
    Purpose: Check that a missing voice file raises FileNotFoundError naming the path and the
        download command, loads nothing, and leaves the cache empty so a later call can retry.
    Parameters:
        fake_piper (type): Fake voice class fixture.
        settings (Settings): Test settings fixture.
        tmp_path (Path): pytest temporary directory.
    Returns:
        None: The test passes or fails through assertions.
    """
    s = settings.model_copy(
        update={"tts": settings.tts.model_copy(update={"voice_dir": tmp_path, "voice": "none"})}
    )
    with pytest.raises(FileNotFoundError) as err:
        get_tts(s)
    msg = str(err.value)
    assert "none.onnx" in msg and "scripts.download_models" in msg
    assert fake_piper.loaded == [] and tts_module._tts is None


def _wav(samples: int, rate: int = 16000, channels: int = 1) -> bytes:
    """
    Function name: _wav
    Purpose: Build a WAV file of silent 16-bit samples for the audio helper tests.
    Parameters:
        samples (int): Number of frames.
        rate (int): Sample rate in Hz.
        channels (int): Number of channels.
    Returns:
        bytes: The WAV file.
    """
    return pcm_to_wav(b"\x00\x00" * samples * channels, rate, channels)


def test_pcm_to_wav_and_read_wav_round_trip():
    """
    Function name: test_pcm_to_wav_and_read_wav_round_trip
    Purpose: Check that PCM data, rate, channels and sample width survive a write and read.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    pcm = bytes(range(16))
    assert read_wav(pcm_to_wav(pcm, 8000, 2, 2)) == (pcm, 8000, 2, 2)
    assert read_wav(pcm_to_wav(pcm, 44100, 1, 1)) == (pcm, 44100, 1, 1)


def test_wav_duration_in_seconds():
    """
    Function name: test_wav_duration_in_seconds
    Purpose: Check the duration of a WAV file: frames divided by the sample rate, also for a
        stereo file.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert wav_duration_s(_wav(16000)) == 1.0
    assert wav_duration_s(_wav(8000, 16000, 2)) == 0.5
    assert wav_duration_s(_wav(0)) == 0.0


def test_concat_wav_empty_list_gives_empty_bytes():
    """
    Function name: test_concat_wav_empty_list_gives_empty_bytes
    Purpose: Check that joining no files gives b"".
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert concat_wav([]) == b""


def test_concat_wav_single_file_has_no_gap():
    """
    Function name: test_concat_wav_single_file_has_no_gap
    Purpose: Check that one file is returned with the same samples and no silence added.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    wav = pcm_to_wav(b"\x05\x00" * 100, 16000)
    assert read_wav(concat_wav([wav])) == read_wav(wav)


def test_concat_wav_inserts_silence_between_files():
    """
    Function name: test_concat_wav_inserts_silence_between_files
    Purpose: Check that silence of the given length is inserted between files only (not at
        the ends), and that a zero length adds none.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    a = pcm_to_wav(b"\x01\x00" * 160, 16000)
    b = pcm_to_wav(b"\x02\x00" * 160, 16000)
    pcm, rate, channels, width = read_wav(concat_wav([a, b, a], silence_ms=100))
    gap = b"\x00\x00" * 1600
    assert (rate, channels, width) == (16000, 1, 2)
    assert pcm == (b"\x01\x00" * 160 + gap + b"\x02\x00" * 160 + gap + b"\x01\x00" * 160)
    assert len(read_wav(concat_wav([a, b], silence_ms=0))[0]) == 640


def test_concat_wav_gap_follows_channel_count():
    """
    Function name: test_concat_wav_gap_follows_channel_count
    Purpose: Check that the silence is sized for the channel count (two bytes per sample per
        channel) so a stereo file stays aligned.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    stereo = _wav(100, 10000, 2)
    pcm, rate, channels, _ = read_wav(concat_wav([stereo, stereo], silence_ms=100))
    assert (rate, channels) == (10000, 2)
    assert len(pcm) == 2 * 100 * 2 * 2 + 1000 * 2 * 2
