from seshat.core.utils.audio import audio_duration_seconds


class AudioValidationError(Exception):
    pass


class FileTooLargeError(AudioValidationError):
    pass


class UnsupportedFormatError(AudioValidationError):
    pass


class AudioValidator:
    @staticmethod
    def check_size(actual_bytes: int, max_bytes: int) -> None:
        if actual_bytes > max_bytes:
            raise FileTooLargeError(f"File size {actual_bytes} exceeds maximum {max_bytes} bytes")

    @staticmethod
    def check_duration(actual_seconds: float, max_seconds: int) -> None:
        if actual_seconds > max_seconds:
            raise AudioValidationError(f"Audio duration {actual_seconds}s exceeds maximum {max_seconds}s")

    @staticmethod
    def validate_magic_bytes(data: bytes, alleged_ext: str | None = None) -> str:
        """Infer file extension from magic bytes; optionally cross-check alleged_ext.

        Never trusts the caller-supplied name — always returns the inferred extension.
        """
        header = data[:12]

        if header.startswith(b"ID3") or header.startswith(b"\xff\xfb") or header.startswith(b"\xff\xf3"):
            inferred = "mp3"
        elif header.startswith(b"RIFF") and header[8:12] == b"WAVE":
            inferred = "wav"
        elif len(header) >= 11 and header[4:11] == b"ftypM4A":
            inferred = "m4a"
        else:
            raise UnsupportedFormatError(f"Unsupported audio format. Magic bytes: {header[:8].hex()}")

        if alleged_ext is not None:
            normalised = alleged_ext.lstrip(".").lower()
            if normalised != inferred:
                raise UnsupportedFormatError(
                    f"Extension mismatch: file claims {normalised!r} but magic bytes indicate {inferred!r}"
                )

        return inferred

    @staticmethod
    def get_duration_seconds(audio_bytes: bytes) -> float:
        duration = audio_duration_seconds(audio_bytes)
        if duration is None:
            raise AudioValidationError("Unable to determine audio duration")
        return duration
