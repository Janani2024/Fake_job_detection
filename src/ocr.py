import numpy as np
from PIL import Image
import io

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], verbose=False)
    return _reader


def extract_text_from_image(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    reader = _get_reader()
    results = reader.readtext(img_array, detail=0, paragraph=True)
    return " ".join(results).strip()
