"""
Tests for the Gemma 4 chat format additions introduced in this PR:
  - format_gemma4() registered formatter
  - Gemma4ChatHandler (CHAT_FORMAT template + get_image_urls() static method)
  - NanoLlavaChatHandler CHAT_FORMAT template
  - MultimodalGemmaChatHandler CHAT_FORMAT template
  - __init__.__version__ value

All tests are pure-Python and do not require a compiled shared library or
an actual model file.
"""

import warnings

import jinja2
import pytest

import llama_cpp
import llama_cpp.llama_chat_format as llama_chat_format
from llama_cpp.llama_chat_format import (
    Gemma4ChatHandler,
    MultimodalGemmaChatHandler,
    NanoLlavaChatHandler,
    format_gemma4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render(chat_format: str, messages, add_generation_prompt: bool = True) -> str:
    """Render a Jinja2 CHAT_FORMAT string with the given messages."""
    env = jinja2.Environment(undefined=jinja2.Undefined, keep_trailing_newline=True)
    tmpl = env.from_string(chat_format)
    return tmpl.render(messages=messages, add_generation_prompt=add_generation_prompt)


# ===========================================================================
# format_gemma4()
# ===========================================================================


class TestFormatGemma4:
    """Tests for the format_gemma4 registered chat formatter."""

    def test_registered_as_gemma4(self):
        """'gemma4' must be a registered chat completion handler."""
        # Access the registry directly; get_chat_formats() is not a public API
        registry = llama_chat_format.LlamaChatCompletionHandlerRegistry()
        assert "gemma4" in registry._chat_handlers

    def test_basic_user_message(self):
        """Single user message produces correct BOS + turn tokens."""
        messages = [{"role": "user", "content": "Hello"}]
        resp = format_gemma4(messages=messages)
        assert resp.prompt.startswith("<bos>")
        assert "<|turn>user\nHello<turn|>\n" in resp.prompt
        # Generation prompt is appended
        assert resp.prompt.endswith("<|turn>model\n")

    def test_system_message_in_thought_channel(self):
        """System messages are wrapped in <|channel>thought…<channel|>."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]
        resp = format_gemma4(messages=messages)
        assert (
            "<|channel>thought\nYou are a helpful assistant.<channel|>\n" in resp.prompt
        )
        # System message should NOT appear as a regular turn
        assert "<|turn>system" not in resp.prompt

    def test_system_message_skipped_in_turns(self):
        """System messages do not produce a <|turn>system… block."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
        ]
        resp = format_gemma4(messages=messages)
        assert "<|turn>system" not in resp.prompt

    def test_assistant_role_mapped_to_model(self):
        """The 'assistant' role must be mapped to 'model' in the prompt."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
            {"role": "user", "content": "How are you?"},
        ]
        resp = format_gemma4(messages=messages)
        assert "<|turn>model\nHello there!<turn|>\n" in resp.prompt
        assert "<|turn>assistant" not in resp.prompt

    def test_multi_turn_conversation(self):
        """Multi-turn conversation produces interleaved user/model turns."""
        messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        resp = format_gemma4(messages=messages)
        assert "<|turn>user\nQuestion 1<turn|>\n" in resp.prompt
        assert "<|turn>model\nAnswer 1<turn|>\n" in resp.prompt
        assert "<|turn>user\nQuestion 2<turn|>\n" in resp.prompt

    def test_stop_tokens(self):
        """format_gemma4 returns the expected stop token list."""
        messages = [{"role": "user", "content": "Hi"}]
        resp = format_gemma4(messages=messages)
        assert isinstance(resp.stop, list)
        assert "<turn|>\n" in resp.stop
        assert "<channel|>" in resp.stop

    def test_empty_system_message_omitted_from_thought_channel(self):
        """An empty string system message produces no thought channel block."""
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hi"},
        ]
        resp = format_gemma4(messages=messages)
        # _get_system_message returns "" → falsy → no channel block inserted
        assert "<|channel>thought" not in resp.prompt

    def test_no_system_message_no_thought_channel(self):
        """Without a system message there should be no thought channel."""
        messages = [{"role": "user", "content": "Hi"}]
        resp = format_gemma4(messages=messages)
        assert "<|channel>thought" not in resp.prompt

    def test_reasoning_budget_accepted(self):
        """reasoning_budget kwarg must not raise."""
        messages = [{"role": "user", "content": "Hi"}]
        resp = format_gemma4(messages=messages, reasoning_budget=512)
        assert resp.prompt  # non-empty

    def test_bos_token_present(self):
        """<bos> is always the very first token."""
        messages = [{"role": "user", "content": "Test"}]
        resp = format_gemma4(messages=messages)
        assert resp.prompt.startswith("<bos>")

    def test_prompt_is_string(self):
        """The returned prompt is always a plain string."""
        messages = [{"role": "user", "content": "Test"}]
        resp = format_gemma4(messages=messages)
        assert isinstance(resp.prompt, str)

    def test_user_role_unchanged(self):
        """The 'user' role is kept as-is (not remapped)."""
        messages = [{"role": "user", "content": "Hello"}]
        resp = format_gemma4(messages=messages)
        assert "<|turn>user\n" in resp.prompt


# ===========================================================================
# Gemma4ChatHandler.get_image_urls()
# ===========================================================================


class TestGemma4ChatHandlerGetImageUrls:
    """Tests for the Gemma4ChatHandler.get_image_urls static method."""

    def test_image_url_as_mapping(self):
        """image_url given as a dict with a 'url' key is extracted correctly."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.jpg"},
                    },
                    {"type": "text", "text": "Describe this"},
                ],
            }
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert urls == ["https://example.com/img.jpg"]

    def test_image_url_as_string(self):
        """image_url given as a plain string is extracted directly."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "https://example.com/photo.png"},
                ],
            }
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert urls == ["https://example.com/photo.png"]

    def test_input_audio_openai_schema(self):
        """OpenAI 'input_audio' content type is converted to a data URI."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"format": "wav", "data": "AUDIO_BASE64"},
                    },
                ],
            }
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert len(urls) == 1
        assert urls[0] == "data:audio/wav;base64,AUDIO_BASE64"

    def test_audio_custom_schema(self):
        """Custom 'audio' content type is also converted to a data URI."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": {"format": "mp3", "data": "MP3_BASE64"},
                    },
                ],
            }
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert len(urls) == 1
        assert urls[0] == "data:audio/mp3;base64,MP3_BASE64"

    def test_mixed_image_and_audio(self):
        """Multiple media items in one message are all extracted in order."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/a.jpg"},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"format": "wav", "data": "WAV64"},
                    },
                    {"type": "text", "text": "Describe and transcribe"},
                ],
            }
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert len(urls) == 2
        assert "https://example.com/a.jpg" in urls
        assert "data:audio/wav;base64,WAV64" in urls

    def test_no_media_returns_empty_list(self):
        """Text-only messages produce an empty list."""
        messages = [
            {"role": "user", "content": "No media here"},
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert urls == []

    def test_non_user_roles_ignored(self):
        """Only 'user' role messages are scanned for media."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/x.jpg"},
                    },
                ],
            }
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert urls == []

    def test_audio_default_format_wav(self):
        """Missing 'format' key in audio dict defaults to 'wav'."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "ABC"},
                        # no 'format' key
                    },
                ],
            }
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        # The default format falls back to "wav" per the implementation
        assert len(urls) == 1
        assert urls[0].startswith("data:audio/wav;base64,")

    def test_multiple_user_messages(self):
        """Media from multiple user messages are all collected."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/1.jpg"},
                    },
                ],
            },
            {"role": "assistant", "content": "OK"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/2.jpg"},
                    },
                ],
            },
        ]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert len(urls) == 2

    def test_string_content_user_message_ignored(self):
        """String-content user messages (no media parts) return empty list."""
        messages = [{"role": "user", "content": "plain text"}]
        urls = Gemma4ChatHandler.get_image_urls(messages)
        assert urls == []


# ===========================================================================
# Gemma4ChatHandler.CHAT_FORMAT template
# ===========================================================================


class TestGemma4ChatHandlerTemplate:
    """Tests for the Gemma4ChatHandler Jinja2 CHAT_FORMAT string."""

    def test_system_message_in_thought_channel(self):
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert "<|channel>thought\nBe concise.<channel|>\n" in prompt

    def test_user_turn_tokens(self):
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [{"role": "user", "content": "Hello"}],
        )
        assert "<|turn>user\nHello<turn|>\n" in prompt

    def test_assistant_turn_tokens(self):
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            add_generation_prompt=False,
        )
        assert "<|turn>model\nA<turn|>\n" in prompt

    def test_generation_prompt_appended(self):
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
        )
        assert prompt.endswith("<|turn>model\n")

    def test_no_generation_prompt_when_false(self):
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=False,
        )
        assert not prompt.endswith("<|turn>model\n")

    def test_image_url_mapping_embedded(self):
        """image_url as a mapping (dict with 'url') is embedded in the prompt."""
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/pic.jpg"},
                        },
                        {"type": "text", "text": "Describe"},
                    ],
                }
            ],
        )
        assert "https://example.com/pic.jpg" in prompt
        assert "Describe" in prompt

    def test_image_url_string_embedded(self):
        """image_url as a plain string is embedded in the prompt."""
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/img.png",
                        },
                        {"type": "text", "text": "Caption"},
                    ],
                }
            ],
        )
        assert "https://example.com/img.png" in prompt

    def test_input_audio_data_uri_embedded(self):
        """input_audio content is serialised as a data: URI."""
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"format": "wav", "data": "TESTDATA"},
                        },
                        {"type": "text", "text": "Transcribe"},
                    ],
                }
            ],
        )
        assert "data:audio/wav;base64,TESTDATA" in prompt

    def test_audio_custom_schema_data_uri_embedded(self):
        """audio (custom schema) content is serialised as a data: URI."""
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": {"format": "mp3", "data": "MP3DATA"},
                        },
                        {"type": "text", "text": "Listen"},
                    ],
                }
            ],
        )
        assert "data:audio/mp3;base64,MP3DATA" in prompt

    def test_tool_call_format(self):
        """Tool-call messages produce the expected <|tool_call> block."""
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris"}',
                            }
                        }
                    ],
                }
            ],
            add_generation_prompt=False,
        )
        assert "<|tool_call>call:get_weather" in prompt

    def test_tool_response_format(self):
        """Tool-response messages produce the expected <|tool_response> block."""
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "tool",
                    "name": "get_weather",
                    "content": "Sunny, 22°C",
                }
            ],
            add_generation_prompt=False,
        )
        assert "<|tool_response>response:get_weather" in prompt
        assert "Sunny, 22°C" in prompt

    def test_media_appears_before_text_in_user_message(self):
        """Media tokens must be emitted before text tokens in the prompt."""
        prompt = _render(
            Gemma4ChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "TEXT_PART"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.jpg"},
                        },
                    ],
                }
            ],
        )
        img_pos = prompt.index("https://example.com/a.jpg")
        txt_pos = prompt.index("TEXT_PART")
        assert img_pos < txt_pos, "image URL should appear before text in the prompt"


# ===========================================================================
# Gemma4ChatHandler thread-safety attribute
# ===========================================================================


class TestGemma4ChatHandlerAttributes:
    """Tests for class-level attributes of Gemma4ChatHandler."""

    def test_format_lock_is_thread_lock(self):
        """_format_lock must be a threading.Lock or RLock instance."""
        import threading

        assert isinstance(Gemma4ChatHandler._format_lock, type(threading.Lock()))

    def test_default_system_message_is_none(self):
        assert Gemma4ChatHandler.DEFAULT_SYSTEM_MESSAGE is None

    def test_chat_format_is_string(self):
        assert isinstance(Gemma4ChatHandler.CHAT_FORMAT, str)


# ===========================================================================
# NanoLlavaChatHandler CHAT_FORMAT template
# ===========================================================================


class TestNanoLlavaChatHandlerTemplate:
    """Tests for the NanoLlavaChatHandler Jinja2 CHAT_FORMAT string."""

    def test_default_system_message(self):
        assert NanoLlavaChatHandler.DEFAULT_SYSTEM_MESSAGE == "Answer the question"

    def test_basic_user_message(self):
        prompt = _render(
            NanoLlavaChatHandler.CHAT_FORMAT,
            [{"role": "user", "content": "What is 2+2?"}],
        )
        assert "<|im_start|>user\n" in prompt
        assert "What is 2+2?" in prompt
        assert "<|im_end|>" in prompt

    def test_generation_prompt(self):
        prompt = _render(
            NanoLlavaChatHandler.CHAT_FORMAT,
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
        )
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_system_message_in_chatml_block(self):
        prompt = _render(
            NanoLlavaChatHandler.CHAT_FORMAT,
            [
                {"role": "system", "content": "Answer the question"},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert "<|im_start|>system\n" in prompt
        assert "Answer the question" in prompt

    def test_image_url_string_before_text(self):
        """Image URL (string) is emitted before text in the user block."""
        prompt = _render(
            NanoLlavaChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "https://img.example.com/x.jpg",
                        },
                        {"type": "text", "text": "Describe"},
                    ],
                }
            ],
        )
        assert "https://img.example.com/x.jpg" in prompt
        img_pos = prompt.index("https://img.example.com/x.jpg")
        txt_pos = prompt.index("Describe")
        assert img_pos < txt_pos

    def test_image_url_mapping_before_text(self):
        """Image URL (mapping with 'url' key) is emitted before text."""
        prompt = _render(
            NanoLlavaChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://img.example.com/y.jpg"},
                        },
                        {"type": "text", "text": "Caption"},
                    ],
                }
            ],
        )
        assert "https://img.example.com/y.jpg" in prompt
        img_pos = prompt.index("https://img.example.com/y.jpg")
        txt_pos = prompt.index("Caption")
        assert img_pos < txt_pos

    def test_assistant_message(self):
        prompt = _render(
            NanoLlavaChatHandler.CHAT_FORMAT,
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            add_generation_prompt=False,
        )
        assert "<|im_start|>assistant\nA<|im_end|>" in prompt


# ===========================================================================
# MultimodalGemmaChatHandler CHAT_FORMAT template
# ===========================================================================


class TestMultimodalGemmaChatHandlerTemplate:
    """Tests for the MultimodalGemmaChatHandler CHAT_FORMAT template."""

    def test_default_system_message_is_none(self):
        assert MultimodalGemmaChatHandler.DEFAULT_SYSTEM_MESSAGE is None

    def test_user_turn_tokens(self):
        prompt = _render(
            MultimodalGemmaChatHandler.CHAT_FORMAT,
            [{"role": "user", "content": "Hello"}],
        )
        assert "<start_of_turn>user\n" in prompt
        assert "<end_of_turn>\n" in prompt

    def test_assistant_turn_tokens(self):
        prompt = _render(
            MultimodalGemmaChatHandler.CHAT_FORMAT,
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            add_generation_prompt=False,
        )
        assert "<start_of_turn>model\nA<end_of_turn>\n" in prompt

    def test_generation_prompt(self):
        prompt = _render(
            MultimodalGemmaChatHandler.CHAT_FORMAT,
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
        )
        assert prompt.endswith("<start_of_turn>model\n")

    def test_image_url_mapping_embedded(self):
        prompt = _render(
            MultimodalGemmaChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/gemma_img.jpg"},
                        },
                        {"type": "text", "text": "Describe"},
                    ],
                }
            ],
        )
        assert "https://example.com/gemma_img.jpg" in prompt

    def test_image_url_string_embedded(self):
        prompt = _render(
            MultimodalGemmaChatHandler.CHAT_FORMAT,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/pic.jpg",
                        },
                        {"type": "text", "text": "Look"},
                    ],
                }
            ],
        )
        assert "https://example.com/pic.jpg" in prompt


# ===========================================================================
# __init__.__version__
# ===========================================================================


class TestPackageVersion:
    """Tests for the package version string exposed via __init__.py."""

    def test_version_is_string(self):
        assert isinstance(llama_cpp.__version__, str)

    def test_version_non_empty(self):
        assert llama_cpp.__version__

    def test_version_format(self):
        """Version must follow a major.minor.patch pattern."""
        parts = llama_cpp.__version__.split(".")
        assert len(parts) == 3, (
            f"Expected 3 version parts, got: {llama_cpp.__version__!r}"
        )
        for part in parts:
            assert part.isdigit(), f"Non-numeric version component: {part!r}"
