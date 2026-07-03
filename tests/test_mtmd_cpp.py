"""
Tests for llama_cpp/mtmd_cpp.py changes introduced in this PR:

  - mtmd_caps Structure (inp_vision, inp_audio fields)
  - mtmd_input_text Structure (_fields_ layout)
  - mtmd_decoder_pos Structure (t, x, y, z fields)
  - mtmd_context_params Structure (image_marker AND new media_marker field)
  - MTMD_INPUT_CHUNK_TYPE_* integer constants
  - mtmd_get_audio_sample_rate function signature
  - mtmd_get_audio_bitrate() deprecated wrapper:
      - emits DeprecationWarning
      - delegates to mtmd_get_audio_sample_rate
      - returns its return value unchanged

The shared library is mocked out by tests/conftest.py so no compiled .so
file is required.
"""

import ctypes
import warnings
from unittest.mock import MagicMock, patch

import pytest

import llama_cpp.mtmd_cpp as mtmd_cpp


# ===========================================================================
# Integer constants
# ===========================================================================


class TestMtmdChunkTypeConstants:
    """The MTMD_INPUT_CHUNK_TYPE_* constants must have the correct integer values."""

    def test_text_constant(self):
        assert mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT == 0

    def test_image_constant(self):
        assert mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE == 1

    def test_audio_constant(self):
        assert mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO == 2

    def test_all_constants_are_distinct(self):
        values = [
            mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT,
            mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE,
            mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO,
        ]
        assert len(set(values)) == len(values)

    def test_constants_are_integers(self):
        assert isinstance(mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT, int)
        assert isinstance(mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE, int)
        assert isinstance(mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO, int)


# ===========================================================================
# mtmd_caps Structure
# ===========================================================================


class TestMtmdCapsStructure:
    """Tests for the mtmd_caps ctypes Structure."""

    def test_is_ctypes_structure(self):
        assert issubclass(mtmd_cpp.mtmd_caps, ctypes.Structure)

    def test_has_inp_vision_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_caps._fields_]
        assert "inp_vision" in field_names

    def test_has_inp_audio_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_caps._fields_]
        assert "inp_audio" in field_names

    def test_exactly_two_fields(self):
        assert len(mtmd_cpp.mtmd_caps._fields_) == 2

    def test_inp_vision_is_cbool(self):
        field_map = {f[0]: f[1] for f in mtmd_cpp.mtmd_caps._fields_}
        assert field_map["inp_vision"] is ctypes.c_bool

    def test_inp_audio_is_cbool(self):
        field_map = {f[0]: f[1] for f in mtmd_cpp.mtmd_caps._fields_}
        assert field_map["inp_audio"] is ctypes.c_bool

    def test_instance_creation(self):
        """mtmd_caps can be instantiated without arguments."""
        caps = mtmd_cpp.mtmd_caps()
        # Fields default to False
        assert isinstance(caps.inp_vision, bool)
        assert isinstance(caps.inp_audio, bool)

    def test_field_assignment(self):
        caps = mtmd_cpp.mtmd_caps()
        caps.inp_vision = True
        caps.inp_audio = True
        assert caps.inp_vision is True
        assert caps.inp_audio is True


# ===========================================================================
# mtmd_input_text Structure
# ===========================================================================


class TestMtmdInputTextStructure:
    """Tests for the mtmd_input_text ctypes Structure."""

    def test_is_ctypes_structure(self):
        assert issubclass(mtmd_cpp.mtmd_input_text, ctypes.Structure)

    def test_has_text_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_input_text._fields_]
        assert "text" in field_names

    def test_has_add_special_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_input_text._fields_]
        assert "add_special" in field_names

    def test_has_parse_special_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_input_text._fields_]
        assert "parse_special" in field_names

    def test_exactly_three_fields(self):
        assert len(mtmd_cpp.mtmd_input_text._fields_) == 3

    def test_text_is_char_p(self):
        field_map = {f[0]: f[1] for f in mtmd_cpp.mtmd_input_text._fields_}
        assert field_map["text"] is ctypes.c_char_p

    def test_add_special_is_cbool(self):
        field_map = {f[0]: f[1] for f in mtmd_cpp.mtmd_input_text._fields_}
        assert field_map["add_special"] is ctypes.c_bool

    def test_parse_special_is_cbool(self):
        field_map = {f[0]: f[1] for f in mtmd_cpp.mtmd_input_text._fields_}
        assert field_map["parse_special"] is ctypes.c_bool

    def test_instance_creation(self):
        it = mtmd_cpp.mtmd_input_text()
        it.text = b"Hello"
        it.add_special = True
        it.parse_special = False
        assert it.text == b"Hello"
        assert it.add_special is True
        assert it.parse_special is False


# ===========================================================================
# mtmd_decoder_pos Structure
# ===========================================================================


class TestMtmdDecoderPosStructure:
    """Tests for the mtmd_decoder_pos ctypes Structure."""

    def test_is_ctypes_structure(self):
        assert issubclass(mtmd_cpp.mtmd_decoder_pos, ctypes.Structure)

    def test_has_t_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_decoder_pos._fields_]
        assert "t" in field_names

    def test_has_x_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_decoder_pos._fields_]
        assert "x" in field_names

    def test_has_y_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_decoder_pos._fields_]
        assert "y" in field_names

    def test_has_z_field(self):
        field_names = [f[0] for f in mtmd_cpp.mtmd_decoder_pos._fields_]
        assert "z" in field_names

    def test_exactly_four_fields(self):
        assert len(mtmd_cpp.mtmd_decoder_pos._fields_) == 4

    def test_all_fields_are_uint32(self):
        for _, ctype in mtmd_cpp.mtmd_decoder_pos._fields_:
            assert ctype is ctypes.c_uint32

    def test_instance_assignment(self):
        dp = mtmd_cpp.mtmd_decoder_pos()
        dp.t = 1
        dp.x = 2
        dp.y = 3
        dp.z = 4
        assert dp.t == 1
        assert dp.x == 2
        assert dp.y == 3
        assert dp.z == 4


# ===========================================================================
# mtmd_context_params Structure
# ===========================================================================


class TestMtmdContextParamsStructure:
    """Tests for the mtmd_context_params ctypes Structure, specifically the new
    media_marker field and the kept image_marker field."""

    def _field_names(self):
        return [f[0] for f in mtmd_cpp.mtmd_context_params._fields_]

    def test_is_ctypes_structure(self):
        assert issubclass(mtmd_cpp.mtmd_context_params, ctypes.Structure)

    def test_has_image_marker_field(self):
        """Deprecated image_marker field must still be present for compatibility."""
        assert "image_marker" in self._field_names()

    def test_has_media_marker_field(self):
        """New media_marker field must be present."""
        assert "media_marker" in self._field_names()

    def test_image_marker_is_char_p(self):
        field_map = {f[0]: f[1] for f in mtmd_cpp.mtmd_context_params._fields_}
        assert field_map["image_marker"] is ctypes.c_char_p

    def test_media_marker_is_char_p(self):
        field_map = {f[0]: f[1] for f in mtmd_cpp.mtmd_context_params._fields_}
        assert field_map["media_marker"] is ctypes.c_char_p

    def test_has_use_gpu_field(self):
        assert "use_gpu" in self._field_names()

    def test_has_n_threads_field(self):
        assert "n_threads" in self._field_names()

    def test_has_warmup_field(self):
        assert "warmup" in self._field_names()

    def test_has_flash_attn_type_field(self):
        assert "flash_attn_type" in self._field_names()


# ===========================================================================
# mtmd_get_audio_bitrate() deprecated wrapper
# ===========================================================================


class TestMtmdGetAudioBitrateDeprecated:
    """Tests for the mtmd_get_audio_bitrate() DeprecationWarning wrapper."""

    def test_emits_deprecation_warning(self):
        """Calling mtmd_get_audio_bitrate must issue a DeprecationWarning."""
        fake_ctx = object()
        with patch.object(mtmd_cpp, "mtmd_get_audio_sample_rate", return_value=16000):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mtmd_cpp.mtmd_get_audio_bitrate(fake_ctx)

        assert len(caught) >= 1
        categories = [w.category for w in caught]
        assert DeprecationWarning in categories

    def test_warning_message_mentions_new_name(self):
        """The deprecation message should name the replacement function."""
        with patch.object(mtmd_cpp, "mtmd_get_audio_sample_rate", return_value=0):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mtmd_cpp.mtmd_get_audio_bitrate(None)

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warnings, "Expected at least one DeprecationWarning"
        msg = str(dep_warnings[0].message)
        assert "mtmd_get_audio_sample_rate" in msg

    def test_delegates_to_sample_rate_function(self):
        """mtmd_get_audio_bitrate must call mtmd_get_audio_sample_rate exactly once."""
        fake_ctx = object()
        with patch.object(
            mtmd_cpp, "mtmd_get_audio_sample_rate", return_value=44100
        ) as mock_fn:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                mtmd_cpp.mtmd_get_audio_bitrate(fake_ctx)

        mock_fn.assert_called_once_with(fake_ctx)

    def test_returns_sample_rate_value(self):
        """The return value of mtmd_get_audio_bitrate must equal the sample rate."""
        expected = 22050
        with patch.object(
            mtmd_cpp, "mtmd_get_audio_sample_rate", return_value=expected
        ):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = mtmd_cpp.mtmd_get_audio_bitrate(None)

        assert result == expected

    def test_passes_context_argument_through(self):
        """The ctx argument is forwarded unchanged to mtmd_get_audio_sample_rate."""
        sentinel_ctx = object()
        with patch.object(
            mtmd_cpp, "mtmd_get_audio_sample_rate", return_value=8000
        ) as mock_fn:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                mtmd_cpp.mtmd_get_audio_bitrate(sentinel_ctx)

        args, _ = mock_fn.call_args
        assert args[0] is sentinel_ctx

    def test_minus_one_propagated_for_no_audio(self):
        """A return value of -1 (no audio support) is passed through intact."""
        with patch.object(mtmd_cpp, "mtmd_get_audio_sample_rate", return_value=-1):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = mtmd_cpp.mtmd_get_audio_bitrate(None)

        assert result == -1


# ===========================================================================
# mtmd_get_audio_sample_rate function exists
# ===========================================================================


class TestMtmdGetAudioSampleRate:
    """Smoke-tests for the new mtmd_get_audio_sample_rate function."""

    def test_function_exists(self):
        """mtmd_get_audio_sample_rate must be defined in the module."""
        assert hasattr(mtmd_cpp, "mtmd_get_audio_sample_rate")
        assert callable(mtmd_cpp.mtmd_get_audio_sample_rate)
