"""Short-lived generated-image artifacts for clients without native image events."""

from __future__ import annotations

import base64
import binascii
import re
import threading
import time
import uuid
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from app.adapters.imagegen import ImageGenerationResult
from app.config import get_config, get_default


_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
_ARTIFACT_RE = re.compile(r"^(?P<token>[0-9a-f]{32})\.(?:png|jpg|webp|gif)$")
_MIME_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}
_lock = threading.Lock()


@dataclass(frozen=True)
class StoredImageResult:
    token: str
    path: Path
    mime_type: str


def image_result_directory() -> Path:
    """Resolve the artifact directory relative to config.json by default."""
    config = get_config()
    configured = str(config.config.get("image_result_dir") or "generated_images")
    path = Path(configured)
    if not path.is_absolute():
        path = config.path.parent / path
    return path.resolve()


def _decode_data_uri(result: ImageGenerationResult) -> tuple[bytes, str]:
    value = str(result.data_uri or "")
    match = re.fullmatch(r"data:([^;,]+);base64,(.+)", value, flags=re.DOTALL)
    if not match:
        raise ValueError("generated image result is not a base64 data URI")
    mime_type = match.group(1).lower()
    if mime_type not in _MIME_EXTENSIONS:
        raise ValueError(f"unsupported generated image MIME type: {mime_type}")
    try:
        data = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("generated image result contains invalid base64 data") from None
    if not data:
        raise ValueError("generated image result is empty")
    return data, mime_type


def _preview_settings() -> tuple[int, int, int]:
    dimension = max(256, min(4096, int(get_default("image_preview_max_dimension", 1280))))
    quality = max(40, min(95, int(get_default("image_preview_quality", 82))))
    max_bytes = max(64 * 1024, min(4 * 1024 * 1024, int(get_default("image_preview_max_bytes", 800000))))
    return dimension, quality, max_bytes


def image_preview_data_uri(result: ImageGenerationResult) -> str:
    """Return a bounded JPEG preview while preserving the original result."""
    data, _mime_type = _decode_data_uri(result)
    if not get_default("image_preview_enabled", True):
        return result.data_uri
    try:
        with Image.open(BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((_preview_settings()[0], _preview_settings()[0]), Image.Resampling.LANCZOS)
            _dimension, quality, max_bytes = _preview_settings()
            encoded = b""
            for current_quality in range(quality, 34, -8):
                output = BytesIO()
                image.save(output, format="JPEG", quality=current_quality, optimize=True, progressive=True)
                encoded = output.getvalue()
                if len(encoded) <= max_bytes:
                    break
            if encoded and len(encoded) <= max_bytes:
                return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    except (OSError, ValueError):
        # Keep compatibility with providers/tests that return non-decodable
        # placeholder bytes; the original artifact remains available.
        pass
    return result.data_uri


def _cleanup(directory: Path, *, now: float, preserve: set[Path] | None = None) -> None:
    ttl_seconds = max(60, int(get_default("image_result_ttl_seconds", 86400)))
    max_files = max(1, int(get_default("image_result_max_files", 500)))
    preserve = preserve or set()
    artifacts: list[tuple[float, Path]] = []
    for path in directory.iterdir():
        if not path.is_file() or not _ARTIFACT_RE.fullmatch(path.name):
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if path not in preserve and now - modified > ttl_seconds:
            try:
                path.unlink()
            except OSError:
                pass
            continue
        artifacts.append((modified, path))
    artifacts.sort(key=lambda item: (item[1] in preserve, item[0]), reverse=True)
    keep_count = max(max_files, len(preserve))
    for _, path in artifacts[keep_count:]:
        try:
            path.unlink()
        except OSError:
            pass


def store_image_results(
    results: list[ImageGenerationResult], *, directory: Path | None = None
) -> list[StoredImageResult]:
    """Persist normalized image results under unguessable capability tokens."""
    target = (directory or image_result_directory()).resolve()
    target.mkdir(parents=True, exist_ok=True)
    decoded = [_decode_data_uri(result) for result in results]
    stored: list[StoredImageResult] = []
    with _lock:
        for data, mime_type in decoded:
            token = uuid.uuid4().hex
            path = target / f"{token}.{_MIME_EXTENSIONS[mime_type]}"
            path.write_bytes(data)
            stored.append(StoredImageResult(token=token, path=path, mime_type=mime_type))
        _cleanup(target, now=time.time(), preserve={item.path for item in stored})
    return stored


def find_image_result(token: str, *, directory: Path | None = None) -> StoredImageResult | None:
    """Resolve a capability token without accepting paths or arbitrary filenames."""
    if not _TOKEN_RE.fullmatch(str(token or "")):
        return None
    target = (directory or image_result_directory()).resolve()
    for mime_type, extension in _MIME_EXTENSIONS.items():
        path = target / f"{token}.{extension}"
        if path.is_file():
            return StoredImageResult(token=token, path=path, mime_type=mime_type)
    return None
