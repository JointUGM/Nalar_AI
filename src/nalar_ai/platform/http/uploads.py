from fastapi import UploadFile

from nalar_ai.shared.errors import InvalidInputError, PayloadTooLargeError

_PDF_MAGIC = b"%PDF-"
_MAGIC_WINDOW = 1024


async def read_pdf_upload(upload: UploadFile, max_bytes: int) -> bytes:
    """Read a PDF upload into memory with a hard size limit and a magic-number check."""
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"upload exceeds {max_bytes} bytes", details={"max_bytes": max_bytes}
        )
    if _PDF_MAGIC not in data[:_MAGIC_WINDOW]:
        raise InvalidInputError("uploaded file is not a PDF")
    return data
