import tiktoken

_encoder = None


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Accurate token count using tiktoken."""
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.encoding_for_model(model)
        except Exception:
            _encoder = tiktoken.get_encoding("cl100k_base")
    return len(_encoder.encode(text))
