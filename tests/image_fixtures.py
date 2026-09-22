"""测试用的真实图片载荷。

网关只提升"能被证明完整的位图"，因此测试数据必须是真实可解码的图片，
不能再用"把一段 base64 重复若干遍凑长度"的写法。
"""
import base64
import struct
import zlib


def png_bytes(width: int = 16, height: int = 16, rgb: tuple = (255, 0, 0)) -> bytes:
    """构造一个合法的 RGB PNG（以 IEND + 其 CRC 结尾）。"""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def png_b64(width: int = 16, height: int = 16) -> str:
    return base64.b64encode(png_bytes(width, height)).decode("ascii")


def png_data_uri(width: int = 16, height: int = 16) -> str:
    return f"data:image/png;base64,{png_b64(width, height)}"


# 1x1 灰度 JPEG：FFD8 开头、FFD9 结尾，base64 长度 216（>100 字符门槛）。
JPEG_B64 = ("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
            "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAA"
            "AAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==")


def jpeg_data_uri() -> str:
    return f"data:image/jpeg;base64,{JPEG_B64}"


def bmp_bytes(total: int = 1200) -> bytes:
    """bfSize 自洽的 BMP：网关靠这个长度字段判断完整性。"""
    return b"BM" + struct.pack("<I", total) + b"\x00" * 4 + b"\x00" * (total - 10)


def bmp_b64(total: int = 1200) -> str:
    """BM 魔数 + 自洽 bfSize（无结束标记的格式靠长度字段证明完整）。"""
    return base64.b64encode(bmp_bytes(total)).decode("ascii")


def webp_bytes() -> bytes:
    """RIFF size 自洽的 WebP（容器字段就足够判完整）。"""
    chunk_body = b"\x00" * 400
    chunk = b"VP8 " + struct.pack("<I", len(chunk_body)) + chunk_body
    body = b"WEBP" + chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


def webp_b64() -> str:
    return base64.b64encode(webp_bytes()).decode("ascii")


def tiff_bytes() -> bytes:
    """TIFF 既无结束标记也无总长度字段，网关无法证明其完整。"""
    return b"II*\x00" + b"\x00" * 1200


def tiff_b64() -> str:
    return base64.b64encode(tiff_bytes()).decode("ascii")


def truncated(payload: str, keep: int = 120, align: bool = True) -> str:
    """模拟被工具输出上限从中间切断的载荷。

    align=True 将长度对齐到 4 的倍数，专门考察“容器结束标记”这一层；
    align=False 保留非法长度，考察 base64 长度校验。
    """
    cut = min(keep, len(payload) - 1)
    if align:
        cut -= cut % 4
    return payload[:cut]


SVG_DATA_URI = ("data:image/svg+xml;base64,"
                + base64.b64encode(
                    b'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
                    b'<rect width="64" height="64" fill="blue"/></svg>').decode("ascii"))
