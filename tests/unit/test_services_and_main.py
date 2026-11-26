"""
File: tests/unit/test_services_and_main.py

Description:
    Unit tests for api/services.py (lazy creation, caching, reload, ingest, status and
    warm-up) and api/main.py (no-cache static files and the warm-up in the lifespan). All
    heavy components are replaced with fakes, so no model is loaded.

Libraries used:
    logging - standard library; handler that collects the services warning messages
    sys - standard library; places a fake ingest.build module in the module cache
    types - standard library; builds the fake module
    pytest - third party; fixtures and monkeypatching
    fastapi.testclient - third party; HTTP client and application lifespan runner
    api.main - project; NoCacheStaticFiles and create_app under test
    api.services - project; the Services class under test
    rag.pipeline - project; build_pipeline is replaced by a counter
    voice.stt - project; get_stt is replaced by a counter
    voice.tts - project; get_tts is replaced by a counter

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

import logging
import sys
import types

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.main import NoCacheStaticFiles, create_app
from api.services import Services


class Counter:
    """
    Class name: Counter
    Purpose: Callable that counts its calls and returns a new object each time, to prove that
        a component is created once and then cached.
    """

    def __init__(self, label: str):
        """
        Function name: Counter.__init__
        Purpose: Store the label and start the call count at zero.
        Parameters:
            label (str): Name used in the created objects.
        Returns:
            None: Initialises the instance.
        """
        self.label = label
        self.calls = 0
        self.args = []

    def __call__(self, *args):
        """
        Function name: Counter.__call__
        Purpose: Count the call, remember the arguments and return a new object.
        Parameters:
            args (tuple): Positional arguments, remembered for the test.
        Returns:
            object: A new object tagged with the label and the call number.
        """
        self.calls += 1
        self.args.append(args)
        return types.SimpleNamespace(label=self.label, n=self.calls)


class FakeRetriever:
    """
    Class name: FakeRetriever
    Purpose: Retriever double that records retrieve() calls and holds a fixed chunk list.
    """

    def __init__(self, fail: bool = False):
        """
        Function name: FakeRetriever.__init__
        Purpose: Store the failure flag and start an empty call log.
        Parameters:
            fail (bool): When True, retrieve() raises RuntimeError.
        Returns:
            None: Initialises the instance.
        """
        self.chunks = [1, 2, 3]
        self.fail = fail
        self.calls = []

    def retrieve(self, query, mode="full", top_k=None):
        """
        Function name: FakeRetriever.retrieve
        Purpose: Record the call, or raise when the failure flag is set.
        Parameters:
            query (str): The search query.
            mode (str): Retrieval mode.
            top_k (int | None): Number of results.
        Returns:
            list: Always an empty list.
        """
        self.calls.append((query, mode, top_k))
        if self.fail:
            raise RuntimeError("retrieve failed")
        return []


class FakeSynth:
    """
    Class name: FakeSynth
    Purpose: Text-to-speech double that records the texts it is asked to speak.
    """

    def __init__(self, fail: bool = False):
        """
        Function name: FakeSynth.__init__
        Purpose: Store the failure flag and start an empty list of texts.
        Parameters:
            fail (bool): When True, synthesize() raises RuntimeError.
        Returns:
            None: Initialises the instance.
        """
        self.fail = fail
        self.spoken = []

    def synthesize(self, text):
        """
        Function name: FakeSynth.synthesize
        Purpose: Record the text, or raise when the failure flag is set.
        Parameters:
            text (str): Text to speak.
        Returns:
            bytes: Always b"wav".
        """
        self.spoken.append(text)
        if self.fail:
            raise RuntimeError("synth failed")
        return b"wav"


@pytest.fixture
def svc(settings):
    """
    Function name: svc
    Purpose: Fixture that provides a new, empty Services object.
    Parameters:
        settings (Settings): Test settings fixture.
    Returns:
        Services: Services with no component loaded.
    """
    return Services(settings)


@pytest.fixture
def service_logs():
    """
    Function name: service_logs
    Purpose: Fixture that collects the messages logged by api.services at warning level or
        above. A handler on that logger is used because create_app replaces the root handlers.
    Parameters:
        None
    Returns:
        list[str]: The formatted messages, filled while the test runs.
    """
    messages: list[str] = []

    class Collect(logging.Handler):
        """
        Class name: Collect
        Purpose: Logging handler that appends each formatted message to a list.
        """

        def emit(self, record):
            """
            Function name: Collect.emit
            Purpose: Store the formatted message of one log record.
            Parameters:
                record (LogRecord): The log record.
            Returns:
                None
            """
            messages.append(record.getMessage())

    logger = logging.getLogger("api.services")
    handler = Collect()
    logger.addHandler(handler)
    yield messages
    logger.removeHandler(handler)


def test_new_services_have_nothing_loaded(svc):
    """
    Function name: test_new_services_have_nothing_loaded
    Purpose: Check that a new Services object holds no pipeline, STT or TTS and that status()
        reports this without loading anything.
    Parameters:
        svc (Services): Empty services fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert svc._pipeline is None and svc._stt is None and svc._tts is None
    status = svc.status()
    assert status["pipeline_loaded"] is False
    assert status["stt_loaded"] is False and status["tts_loaded"] is False
    assert "chunks" not in status and "llm_model" not in status
    assert status["llm_provider"] == svc.settings.llm.provider
    assert status["vector_store"] == "faiss"


def test_pipeline_is_built_once_with_settings(svc, monkeypatch):
    """
    Function name: test_pipeline_is_built_once_with_settings
    Purpose: Check that pipeline() calls build_pipeline once with the settings and then
        returns the cached object.
    Parameters:
        svc (Services): Empty services fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    build = Counter("pipeline")
    monkeypatch.setattr("rag.pipeline.build_pipeline", build)
    first = svc.pipeline()
    assert svc.pipeline() is first and build.calls == 1
    assert build.args == [(svc.settings,)]


def test_stt_is_built_once_with_settings(svc, monkeypatch):
    """
    Function name: test_stt_is_built_once_with_settings
    Purpose: Check that stt() calls get_stt once with the settings and caches the result.
    Parameters:
        svc (Services): Empty services fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    build = Counter("stt")
    monkeypatch.setattr("voice.stt.get_stt", build)
    first = svc.stt()
    assert svc.stt() is first and build.calls == 1
    assert build.args == [(svc.settings,)]
    assert svc.status()["stt_loaded"] is True


def test_tts_is_built_once_with_settings(svc, monkeypatch):
    """
    Function name: test_tts_is_built_once_with_settings
    Purpose: Check that tts() calls get_tts once with the settings and caches the result.
    Parameters:
        svc (Services): Empty services fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    build = Counter("tts")
    monkeypatch.setattr("voice.tts.get_tts", build)
    first = svc.tts()
    assert svc.tts() is first and build.calls == 1
    assert build.args == [(svc.settings,)]
    assert svc.status()["tts_loaded"] is True


def test_failed_build_is_not_cached(svc, monkeypatch):
    """
    Function name: test_failed_build_is_not_cached
    Purpose: Check that when get_tts raises, the error reaches the caller and the next call
        tries again.
    Parameters:
        svc (Services): Empty services fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    attempts = []

    def flaky(settings):
        """
        Function name: flaky
        Purpose: Raise on the first call and return an object on the second.
        Parameters:
            settings (Settings): Settings, ignored.
        Returns:
            object: A simple namespace on the second call.
        """
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("voice missing")
        return types.SimpleNamespace(ok=True)

    monkeypatch.setattr("voice.tts.get_tts", flaky)
    with pytest.raises(FileNotFoundError):
        svc.tts()
    assert svc._tts is None
    assert svc.tts().ok is True and len(attempts) == 2


def test_reload_index_drops_only_the_pipeline(svc, monkeypatch):
    """
    Function name: test_reload_index_drops_only_the_pipeline
    Purpose: Check that reload_index() resets the pipeline so the next pipeline() call builds
        a new one, and leaves STT and TTS untouched.
    Parameters:
        svc (Services): Empty services fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    build = Counter("pipeline")
    monkeypatch.setattr("rag.pipeline.build_pipeline", build)
    svc._stt, svc._tts = object(), object()
    stt, tts = svc._stt, svc._tts
    first = svc.pipeline()
    svc.reload_index()
    assert svc._pipeline is None
    second = svc.pipeline()
    assert second is not first and second.n == 2 and build.calls == 2
    assert svc._stt is stt and svc._tts is tts


def test_reload_index_on_empty_services_is_safe(svc):
    """
    Function name: test_reload_index_on_empty_services_is_safe
    Purpose: Check that reload_index() works when nothing is loaded yet.
    Parameters:
        svc (Services): Empty services fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    svc.reload_index()
    assert svc._pipeline is None


def test_ingest_builds_index_and_reloads(svc, monkeypatch):
    """
    Function name: test_ingest_builds_index_and_reloads
    Purpose: Check that ingest() calls build_index with the settings, returns its manifest
        and drops the cached pipeline.
    Parameters:
        svc (Services): Empty services fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    seen = []

    def fake_build(settings):
        """
        Function name: fake_build
        Purpose: Record the settings and return a fixed manifest.
        Parameters:
            settings (Settings): Settings given by Services.ingest.
        Returns:
            dict: The manifest.
        """
        seen.append(settings)
        return {"chunks": 12}

    module = types.ModuleType("ingest.build")
    module.build_index = fake_build
    monkeypatch.setitem(sys.modules, "ingest.build", module)
    svc._pipeline = object()
    assert svc.ingest() == {"chunks": 12}
    assert seen == [svc.settings] and svc._pipeline is None


def test_ingest_error_keeps_the_pipeline(svc, monkeypatch):
    """
    Function name: test_ingest_error_keeps_the_pipeline
    Purpose: Check that when build_index raises, the error is passed on and the current
        pipeline is kept.
    Parameters:
        svc (Services): Empty services fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """

    def failing_build(settings):
        """
        Function name: failing_build
        Purpose: Raise FileNotFoundError as an empty docs folder would.
        Parameters:
            settings (Settings): Settings, ignored.
        Returns:
            dict: never returns; always raises FileNotFoundError.
        """
        raise FileNotFoundError("no docs")

    module = types.ModuleType("ingest.build")
    module.build_index = failing_build
    monkeypatch.setitem(sys.modules, "ingest.build", module)
    pipeline = object()
    svc._pipeline = pipeline
    with pytest.raises(FileNotFoundError):
        svc.ingest()
    assert svc._pipeline is pipeline


def test_status_with_pipeline_reports_chunks_and_llm(svc):
    """
    Function name: test_status_with_pipeline_reports_chunks_and_llm
    Purpose: Check that status() adds the chunk count, the LLM model and the reachability
        flag from health() when the pipeline is loaded.
    Parameters:
        svc (Services): Empty services fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    llm = types.SimpleNamespace(model="m1", health=lambda: 1)
    svc._pipeline = types.SimpleNamespace(retriever=FakeRetriever(), llm=llm)
    status = svc.status()
    assert status["pipeline_loaded"] is True
    assert status["chunks"] == 3 and status["llm_model"] == "m1"
    assert status["llm_reachable"] is True


def test_status_reachable_is_none_without_health(svc):
    """
    Function name: test_status_reachable_is_none_without_health
    Purpose: Check that llm_reachable is None when the LLM has no health method and False
        when health() returns a false value.
    Parameters:
        svc (Services): Empty services fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    svc._pipeline = types.SimpleNamespace(
        retriever=FakeRetriever(), llm=types.SimpleNamespace(model="m")
    )
    assert svc.status()["llm_reachable"] is None
    svc._pipeline = types.SimpleNamespace(
        retriever=FakeRetriever(), llm=types.SimpleNamespace(model="m", health=lambda: False)
    )
    assert svc.status()["llm_reachable"] is False


def _patch_components(svc, retriever, tts, stt_fail=False):
    """
    Function name: _patch_components
    Purpose: Replace the pipeline, stt and tts methods of the services with fakes that record
        their calls, and return the list of the component names that were created.
    Parameters:
        svc (Services): Services to patch.
        retriever (FakeRetriever): Retriever of the fake pipeline.
        tts (FakeSynth): Text-to-speech double.
        stt_fail (bool): When True, stt() raises RuntimeError.
    Returns:
        list[str]: Names appended in creation order.
    """
    created = []

    def pipeline():
        """
        Function name: pipeline
        Purpose: Record the call and return a fake pipeline holding the retriever.
        Parameters:
            None
        Returns:
            SimpleNamespace: The fake pipeline.
        """
        created.append("pipeline")
        return types.SimpleNamespace(retriever=retriever)

    def stt():
        """
        Function name: stt
        Purpose: Record the call and raise when the failure flag is set.
        Parameters:
            None
        Returns:
            object: A plain object.
        """
        created.append("stt")
        if stt_fail:
            raise RuntimeError("no whisper model")
        return object()

    def get_tts():
        """
        Function name: get_tts
        Purpose: Record the call and return the text-to-speech double.
        Parameters:
            None
        Returns:
            FakeSynth: The double.
        """
        created.append("tts")
        return tts

    svc.pipeline, svc.stt, svc.tts = pipeline, stt, get_tts
    return created


def test_warm_up_loads_components_and_runs_inference(svc):
    """
    Function name: test_warm_up_loads_components_and_runs_inference
    Purpose: Check that warm_up() creates the pipeline, STT and TTS, retrieves once with the
        query "warm up" and speaks "Ready.".
    Parameters:
        svc (Services): Empty services fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    retriever, tts = FakeRetriever(), FakeSynth()
    created = _patch_components(svc, retriever, tts)
    svc.warm_up()
    assert created[:3] == ["pipeline", "stt", "tts"]
    assert retriever.calls == [("warm up", "full", 1)]
    assert tts.spoken == ["Ready."]


def test_warm_up_survives_a_missing_component(svc, service_logs):
    """
    Function name: test_warm_up_survives_a_missing_component
    Purpose: Check that when STT fails to load, warm_up() logs a warning naming the component
        and still loads the other components and runs the inference.
    Parameters:
        svc (Services): Empty services fixture.
        service_logs (list[str]): Messages logged by api.services.
    Returns:
        None: The test passes or fails through assertions.
    """
    retriever, tts = FakeRetriever(), FakeSynth()
    created = _patch_components(svc, retriever, tts, stt_fail=True)
    svc.warm_up()
    assert "tts" in created and tts.spoken == ["Ready."]
    assert any("component=stt failed" in m and "no whisper model" in m for m in service_logs)


def test_warm_up_logs_inference_failure(svc, service_logs):
    """
    Function name: test_warm_up_logs_inference_failure
    Purpose: Check that an error during the warm-up retrieval is logged and not raised, and
        that the synthesis is then skipped.
    Parameters:
        svc (Services): Empty services fixture.
        service_logs (list[str]): Messages logged by api.services.
    Returns:
        None: The test passes or fails through assertions.
    """
    tts = FakeSynth()
    _patch_components(svc, FakeRetriever(fail=True), tts)
    svc.warm_up()
    assert tts.spoken == []
    assert any("warm_up inference failed: retrieve failed" in m for m in service_logs)


def test_warm_up_logs_synthesis_failure(svc, service_logs):
    """
    Function name: test_warm_up_logs_synthesis_failure
    Purpose: Check that an error while speaking the warm-up text is logged and not raised.
    Parameters:
        svc (Services): Empty services fixture.
        service_logs (list[str]): Messages logged by api.services.
    Returns:
        None: The test passes or fails through assertions.
    """
    _patch_components(svc, FakeRetriever(), FakeSynth(fail=True))
    svc.warm_up()
    assert any("warm_up inference failed: synth failed" in m for m in service_logs)


def test_static_page_has_no_store_cache_header(settings, tmp_path):
    """
    Function name: test_static_page_has_no_store_cache_header
    Purpose: Check that the root page and another static file are served with
        "Cache-Control: no-store".
    Parameters:
        settings (Settings): Test settings fixture.
        tmp_path (Path): pytest temporary directory.
    Returns:
        None: The test passes or fails through assertions.
    """
    client = TestClient(create_app(settings=settings, services=Services(settings)))
    r = client.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    r2 = client.get("/index.html")
    assert r2.headers["cache-control"] == "no-store"


def test_no_cache_static_files_sets_header_on_any_file(tmp_path):
    """
    Function name: test_no_cache_static_files_sets_header_on_any_file
    Purpose: Check that NoCacheStaticFiles adds the no-store header to a file in its own
        directory.
    Parameters:
        tmp_path (Path): pytest temporary directory.
    Returns:
        None: The test passes or fails through assertions.
    """
    from fastapi import FastAPI

    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    app = FastAPI()
    app.mount("/", NoCacheStaticFiles(directory=str(tmp_path)))
    r = TestClient(app).get("/a.txt")
    assert r.status_code == 200 and r.text == "hello"
    assert r.headers["cache-control"] == "no-store"


def test_missing_static_file_is_404(settings):
    """
    Function name: test_missing_static_file_is_404
    Purpose: Check that an unknown static path gives 404 and is not a server error.
    Parameters:
        settings (Settings): Test settings fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    client = TestClient(create_app(settings=settings, services=Services(settings)))
    assert client.get("/no-such-file.js").status_code == 404


def _app_with_warm_up(settings, monkeypatch, warm_up, injected):
    """
    Function name: _app_with_warm_up
    Purpose: Build an application whose Services.warm_up only counts calls, with the given
        warm_up setting, and run its lifespan once.
    Parameters:
        settings (Settings): Test settings.
        monkeypatch (MonkeyPatch): pytest patching helper.
        warm_up (bool): Value of the api.warm_up setting.
        injected (bool): Whether a Services object is passed to create_app.
    Returns:
        list[int]: One entry per warm_up call.
    """
    calls = []
    monkeypatch.setattr(Services, "warm_up", lambda self: calls.append(1))
    s = settings.model_copy(update={"api": settings.api.model_copy(update={"warm_up": warm_up})})
    app = create_app(settings=s, services=Services(s) if injected else None)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    return calls


def test_lifespan_warms_up_when_enabled(settings, monkeypatch):
    """
    Function name: test_lifespan_warms_up_when_enabled
    Purpose: Check that the lifespan calls warm_up once when api.warm_up is true and no
        services were injected.
    Parameters:
        settings (Settings): Test settings fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert _app_with_warm_up(settings, monkeypatch, warm_up=True, injected=False) == [1]


def test_lifespan_skips_warm_up_when_disabled(settings, monkeypatch):
    """
    Function name: test_lifespan_skips_warm_up_when_disabled
    Purpose: Check that the lifespan does not call warm_up when api.warm_up is false.
    Parameters:
        settings (Settings): Test settings fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert _app_with_warm_up(settings, monkeypatch, warm_up=False, injected=False) == []


def test_lifespan_skips_warm_up_for_injected_services(settings, monkeypatch):
    """
    Function name: test_lifespan_skips_warm_up_for_injected_services
    Purpose: Check that the lifespan does not call warm_up when services are injected, even
        when api.warm_up is true.
    Parameters:
        settings (Settings): Test settings fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert _app_with_warm_up(settings, monkeypatch, warm_up=True, injected=True) == []


def test_create_app_uses_injected_services_and_cors(settings):
    """
    Function name: test_create_app_uses_injected_services_and_cors
    Purpose: Check that the injected Services object is stored on the application state, that
        the application title is set and that CORS exposes the X-Request-ID header.
    Parameters:
        settings (Settings): Test settings fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    svc = Services(settings)
    app = create_app(settings=settings, services=svc)
    assert app.state.services is svc and app.title == "Voice RAG Assistant"
    r = TestClient(app).get("/health", headers={"Origin": "http://example.com"})
    assert r.headers["access-control-allow-origin"] == "*"
    assert "X-Request-ID" in r.headers["access-control-expose-headers"]


def test_create_app_without_web_dir_skips_static_mount(settings, monkeypatch, tmp_path):
    """
    Function name: test_create_app_without_web_dir_skips_static_mount
    Purpose: Check that when the web folder does not exist, no static page is mounted and the
        API routes still work.
    Parameters:
        settings (Settings): Test settings fixture.
        monkeypatch (MonkeyPatch): pytest patching helper.
        tmp_path (Path): pytest temporary directory.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.setattr(main_module, "WEB_DIR", tmp_path / "missing")
    client = TestClient(create_app(settings=settings, services=Services(settings)))
    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200
