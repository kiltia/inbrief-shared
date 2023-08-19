import copy
import re

DATE_FORMAT = "%m/%d/%y %H:%M:%S"
DEFAULT_END_DATE = "01/01/01 00:00:00"
DEFAULT_PAYLOAD_STRUCTURE: dict = {
    "channel": [],
    "id": [],
    "text": [],
    "date": [],
}
SOCIAL_FEATURES = ["comments", "reactions"]
LOGGING_FORMAT = "[%(levelname)s] [%(asctime)s] %(message)s"
SESSION_PATH = "sessions"
CACHE_PATH = "cache"


def clean_text(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        re.UNICODE,
    )
    text = re.sub(
        r"https?:\/\/.*[\r\n]*", "", emoji_pattern.sub(r"", text), flags=re.MULTILINE
    )
    text = re.sub(r"@.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"@.*[\r\n]#", "", text, flags=re.MULTILINE)
    return text


def add_optional_columns(default_scheme, embedders, social, markup):
    default_scheme = copy.deepcopy(default_scheme)
    for embedder in embedders:
        default_scheme[embedder.get_label()] = []
    if social:
        for feature in SOCIAL_FEATURES:
            default_scheme[feature] = []
    if markup:
        default_scheme["cls"] = []
    return default_scheme
