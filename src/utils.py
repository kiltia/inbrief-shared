import copy

DEFAULT_PAYLOAD_STRUCTURE: dict = {
    "channel": [],
    "id": [],
    "text": [],
    "date": [],
}
SOCIAL_FEATURES = ["comments", "reactions"]
SESSION_PATH = "sessions"


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
