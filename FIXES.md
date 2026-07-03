# FIXES.md — Gemma 4 merge from `llama-cpp-python-old`

Date: 2026-07-03

This documents the merge of the custom Gemma 4 work (chat handler with vision,
audio, tool calling, and thinking-mode support) from the old working copy
(`C:\AI\llama-cpp-python-old`) onto the current upstream base (v0.3.32,
commit `346853c`). The strategy was a **curated merge**: keep the newer
upstream infrastructure everywhere, and port only the Gemma-specific additions
on top of it. All ported blocks were verified byte-identical to the old
implementations via AST source comparison.

---

## 1. `llama_cpp/llama_chat_format.py` (+364 / −3 lines)

### Line 10 — added import
```python
import threading
```
Required by `Gemma4ChatHandler._format_lock`.

### Line 1438 — comment clarification
The `format_gemma` header comment now reads
"Google's Gemma models (Gemma 2 and Gemma 3)" to distinguish it from the new
Gemma 4 format below it.

### Lines 1458–1529 — added `format_gemma4` (registered as `"gemma4"`)
Text-only prompt formatter for Gemma 4:
- `<bos>` + `<|turn>role\n...<turn|>` turn structure, assistant mapped to `model`.
- System messages rendered into a `<|channel>thought ... <channel|>` block.
- Accepts a `reasoning_budget` keyword (flows in automatically because the
  generic handler forwards `**kwargs` to registered formatters).
- Stop tokens: `<turn|>\n`, `<channel|>`, `<turn|>`.

### Lines 3849–3904 — added `GemmaChatHandler(Llava15ChatHandler)`
Multimodal handler for Gemma 2/3-family vision models (PaliGemma, MedGemma)
using `<start_of_turn>`/`<end_of_turn>` tokens; system messages folded into a
user turn. **This replaces the empty stub the upstream base had at this spot**
(`class Gemma4ChatHandler(MTMDChatHandler): pass`) — the real Gemma 4 handler
now lives at line 4302 (see below).

### Lines 4259–4294 — added `MultimodalGemmaChatHandler(Llava15ChatHandler)`
Minimal `<start_of_turn>`-style multimodal template with no default system
message (Gemma has no native system role).

### Lines 4302–4488 — added full `Gemma4ChatHandler(Llava15ChatHandler)`
The complete Gemma 4 handler ported from the old repo:
- **Line 4316** — `DEFAULT_SYSTEM_MESSAGE = None`.
- **Line 4322** — class-level `_format_lock` (`threading.Lock`) protecting the
  `CHAT_FORMAT` mutation in `__call__` against concurrent requests, so a
  thinking-mode call cannot leak `<|think|>` tokens into a standard call on a
  shared instance.
- **Lines 4324–4375** — `CHAT_FORMAT` Jinja template covering:
  - system messages in `<|channel>thought ... <channel|>`;
  - user turns with media emitted before text — images (`image_url` string or
    mapping) **and audio** (`input_audio` OpenAI schema or custom `audio`
    schema, emitted as `data:audio/<fmt>;base64,...` URIs);
  - assistant turns as `<|turn>model ... <turn|>`;
  - tool calls (`<|tool_call>call:<name><args><tool_call|>`) and tool
    responses (`<|tool_response>response:...<tool_response|>`).
- **Lines 4378–4412** — `get_image_urls()` override that extracts image URLs
  *and* audio base64 payloads so the mtmd backend replaces them with media
  marker embeddings.
- **Lines 4414–4488** — `__call__` override:
  - `enable_thinking=True` injects `<|think|>` into the system thought channel
    and the generation prompt, inserting a blank system message if none exists;
  - clears llama state per call (`reset()`, `kv_cache_clear()`, `n_tokens = 0`,
    `input_ids.fill(0)`, cached image embeds) for reliable multi-turn
    multimodal use, matching the Qwen25VL handler pattern;
  - template mutation + `super().__call__()` + restore runs under
    `_format_lock` with a `finally` restore;
  - non-streaming thinking-mode responses get a `message["thinking"]` field.

### Wiring that now activates (no changes needed — already in the base)
- `llama_cpp/server/model.py:118-123` maps `chat_format in ("mtmd", "gemma4")`
  to `MTMDChatHandler` / `Gemma4ChatHandler`; the latter was an empty stub and
  is now the full implementation.
- README model table entry for `gemma-4` / `Gemma4ChatHandler` / `gemma4`.
- `examples/server/server.py` `gemma4-tool-call` parser and the colab
  notebooks (`Gemma4-12B-QAT.ipynb`, `notebook.ipynb`) — identical in both
  repos already.

---

## 2. `tests/` — three new files

### `tests/test_gemma4_chat_format.py` (729 lines, new)
Ported unchanged from the old repo. Covers `format_gemma4` (registration,
turn/channel tokens, system-in-thought-channel, role mapping, stop tokens,
reasoning_budget), `Gemma4ChatHandler.get_image_urls` (image mapping/string,
OpenAI `input_audio`, custom `audio`, mixed media, defaults), the
`Gemma4ChatHandler` template (turn tokens, media-before-text ordering, tool
call/response rendering, generation prompt), the `_format_lock`, and the
`MultimodalGemmaChatHandler` template.

### `tests/test_mtmd_cpp.py` (332 lines, new)
Ported unchanged from the old repo. Structural tests for the
`llama_cpp.mtmd_cpp` ctypes bindings.

### `tests/conftest.py` (65 lines, new — ported **with fixes**)
Mocks `ctypes.CDLL` and `pathlib.Path.exists` so the suite can import
`llama_cpp` on a source-only checkout (no compiled library). Two fixes were
applied on top of the old repo's version:

- **Lines 18–23** (`_HAS_REAL_LIB`) and **line 48** (`if not _HAS_REAL_LIB:`)
  — the old conftest patched unconditionally, which would have broken the
  real-model integration tests on any machine/CI where the compiled library
  *is* present. The mocks now activate only when `llama_cpp/lib` contains no
  `.so`/`.dylib`/`.dll`.
- **Lines 52–65** (`_patched_add_dll_directory`) — Windows fix. On Windows,
  `load_shared_library()` calls `os.add_dll_directory()` on the (nonexistent)
  `llama_cpp/lib` directory *before* the mocked `CDLL` is ever reached, so the
  old repo's suite could not even collect (`FileNotFoundError` at import).
  Missing directories are now tolerated.

---

## 3. Kept from the new upstream base (old-repo removals **not** merged)

The old file appears to have had a standalone version pasted over newer
upstream code at some point, deleting upstream improvements. These were
deliberately **retained**:

- `MTMDChatHandler` — the generic multimodal handler (old file deleted it,
  which would have broken `chat_format="mtmd"` in `server/model.py`).
- Transformers-aligned Jinja environment in `Jinja2ChatFormatter`: the
  `tojson` filter, the `{% generation %}` tag pass-through extension, and
  `loopcontrols` (upstream #1486, #2018, #2226).
- Streaming logprobs conversion
  (`_convert_text_completion_logprobs_to_chat`) in the functionary handler —
  the old file replaced one call site with `logprobs: None` (a regression).
- `llama_model_n_layer_nextn` binding in `llama_cpp/llama_cpp.py` — the only
  difference in that file; the old copy simply predates it (upstream #2318+).

---

## 4. Deliberately not ported (available on request)

- **Functionary `generate_streaming` chunk-safety refactor** (old commits
  `f391065`, `7e04f11`): `uuid`/`time`-based fallbacks for `chunk_id`/
  `chunk_created`/`chunk_model` guarding against `UnboundLocalError` on empty
  completion streams. Real fix, but it conflicts with upstream's newer
  streaming-logprobs code in the same hunks and included the logprobs
  regression noted above. Can be re-ported cleanly if functionary models are
  in use.
- **`stop: ... = []` → `stop: ... = None` default changes** (5 signatures):
  style-only (the default is never mutated); skipped to minimize divergence
  from upstream.
- **Untracked dev files from the old repo**: `.editorconfig`, `build.txt`,
  `monitor_inference-full.py`.

---

## 5. Verification

- `python -m py_compile llama_cpp/llama_chat_format.py` — OK.
- AST comparison of `format_gemma4`, `GemmaChatHandler`,
  `MultimodalGemmaChatHandler`, `Gemma4ChatHandler` against the old file —
  all byte-identical.
- `pytest tests/test_gemma4_chat_format.py tests/test_mtmd_cpp.py` —
  **100 passed**.
- `pytest tests/` — **107 passed, 19 failed**; all 19 failures are
  pre-existing real-model integration tests in `test_llama.py`
  (`test_real_llama*`, `test_*_prompt_cache*`, `test_*_matches_fresh`,
  `test_llama_cpp_tokenization`) that require the compiled library and
  downloaded GGUF models, which this source-only checkout does not have.
  They are unrelated to the merge and still run for real on a built tree.
