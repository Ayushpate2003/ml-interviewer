# Test Plan — Privacy-First AI Interviewer

## 1. Testing Philosophy for a 1-Day Build
Given the time budget, automated tests focus on the **deterministic, high-risk plumbing** (parsing, DB writes, report generation) rather than the LLM's actual output quality, which is validated manually via live demo rehearsal instead. Mock the Ollama/Gemma 4 call in all automated tests — never depend on a live model response for CI-style checks, since output is non-deterministic and slow.

## 2. Test Priorities
| Priority | Module | Why |
|---|---|---|
| P0 | `llm/parser.py` (JSON extraction) | Single highest risk of a silent failure crashing the report step |
| P0 | `stt/transcribe.py` (empty audio handling) | Silence is a very likely live-demo occurrence |
| P1 | `memory/db.py` (SQLite CRUD) | Report generation depends entirely on correct persistence |
| P1 | `report/generate_report.py` | Must never crash on missing/partial data |
| P2 | `llm/client.py` (Ollama call wrapper) | Mostly a thin HTTP wrapper — light testing, mocked network |
| P2 | `tts/speak.py` | Manual verification (audio output) is more useful than automated assertions here |

## 3. Unit Tests

### 3.1 `llm/parser.py`
```python
def test_parses_clean_json():
    raw = '{"overall_score": 4.0, "dimensions": []}'
    result = parse_score_json(raw)
    assert result["overall_score"] == 4.0

def test_parses_json_wrapped_in_code_fence():
    raw = '```json\n{"overall_score": 3.5, "dimensions": []}\n```'
    result = parse_score_json(raw)
    assert result["overall_score"] == 3.5

def test_parses_json_with_prose_preamble():
    raw = 'Here is the evaluation:\n{"overall_score": 2.0, "dimensions": []}'
    result = parse_score_json(raw)
    assert result["overall_score"] == 2.0

def test_raises_or_falls_back_on_unparseable_output():
    raw = 'Sorry, I cannot provide a score right now.'
    result = parse_score_json(raw, fallback={"overall_score": None, "dimensions": []})
    assert result["overall_score"] is None  # falls back cleanly, doesn't crash
```

### 3.2 `stt/transcribe.py`
```python
def test_empty_audio_returns_empty_string(monkeypatch):
    # feed a near-silent buffer
    result = transcribe(silent_audio_bytes)
    assert result.strip() == ""

def test_transcribe_returns_nonempty_for_known_sample():
    result = transcribe(load_fixture("sample_answer.wav"))
    assert len(result) > 0

def test_handles_unsupported_audio_gracefully():
    with pytest.raises(TranscriptionError):
        transcribe(b"not-real-audio-bytes")
```

### 3.3 `memory/db.py`
```python
def test_create_session_and_append_turns(tmp_db):
    session_id = create_session(tmp_db, role="Backend Engineer")
    add_turn(tmp_db, session_id, speaker="interviewer", content="Tell me about a challenging bug.")
    add_turn(tmp_db, session_id, speaker="candidate", content="I once debugged a race condition...")
    turns = get_turns(tmp_db, session_id)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "interviewer"

def test_scores_persist_and_retrieve(tmp_db):
    session_id = create_session(tmp_db, role="HR Round")
    save_scores(tmp_db, session_id, [{"dimension": "communication_clarity", "score": 4, "justification": "Clear"}])
    scores = get_scores(tmp_db, session_id)
    assert scores[0]["score"] == 4
```

### 3.4 `report/generate_report.py`
```python
def test_report_generates_pdf_file(tmp_path, sample_session):
    output_path = generate_report(sample_session, out_dir=tmp_path)
    assert output_path.exists()
    assert output_path.suffix == ".pdf"

def test_report_handles_missing_scores_gracefully(tmp_path, session_with_no_scores):
    output_path = generate_report(session_with_no_scores, out_dir=tmp_path)
    assert output_path.exists()  # should not crash, should render a partial report

def test_report_includes_full_transcript(tmp_path, sample_session):
    output_path = generate_report(sample_session, out_dir=tmp_path)
    text = extract_pdf_text(output_path)
    assert "Tell me about a challenging bug" in text
```

### 3.5 `llm/client.py` (mocked)
```python
def test_client_sends_expected_payload(mock_ollama):
    call_gemma(history=[{"speaker": "candidate", "content": "..."}], role="Backend Engineer")
    sent_payload = mock_ollama.last_request()
    assert sent_payload["model"] == "gemma4:4b"
    assert "Backend Engineer" in sent_payload["messages"][0]["content"]

def test_client_raises_clear_error_if_ollama_unreachable(mock_ollama_down):
    with pytest.raises(ConnectionError):
        call_gemma(history=[], role="Backend Engineer")
```

## 4. Integration Test (manual, run before every demo rehearsal)
Checklist rather than automated code, since it spans real audio/model latency:
1. [ ] Fresh session, offline (Wi-Fi off) — full 5-turn interview completes without any error dialogs.
2. [ ] Silence on one turn → app recovers with the "didn't catch that" prompt, doesn't crash.
3. [ ] Report PDF downloads and opens correctly, contains all turns and a plausible scorecard.
4. [ ] Total elapsed time for a 5-turn session is under ~5 minutes (demo-video-friendly).
5. [ ] Ollama/model load time measured once at startup — confirm it's not repeated per turn.

## 5. What Is *Not* Automated (and why)
- **Quality of Gemma 4's follow-up questions** — inherently subjective; validated by human read-through during rehearsal, not asserted in test code.
- **TTS audio naturalness** — verified by ear, not by an automated audio-quality metric (out of scope for a 1-day build).
- **Cross-platform mic behavior** — tested manually on the actual demo machine(s) rather than via CI, since browser mic permission behavior varies by OS/browser.
