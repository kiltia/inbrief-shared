from config import DefaultClusteringSettings, DefaultSearcherSettings, LinkerAPISettings
from flask import Flask, make_response, request
from matcher import Matcher

app = Flask(__name__)
app.config["API_SETTINGS"] = LinkerAPISettings()
app.config["clustering"] = DefaultClusteringSettings().model_dump()
app.config["bm25"] = DefaultSearcherSettings().model_dump()


@app.route(app.config["API_SETTINGS"].get_stories, methods=["POST"])
def get_stories():
    data = request.get_json()
    matcher = Matcher(data["texts"], data["embeddings"], data["dates"])
    if (
        "method" in data
        and data["method"] in app.config["API_SETTINGS"].available_methods
    ):
        method = data["method"]
    elif not ("method" in data):
        method = app.config["API_SETTINGS"].default_method
    else:
        return make_response("Specified matching method does not exist!", 204)
    config = app.config[method]
    if "config" in data:
        for i in config:
            if i in data["config"]:
                config[i] = data["config"][i]
            else:
                return make_response("Bad argument", 204)
    stories, num_texts_in_stories = matcher.get_stories(method, **app.config[method])
    return make_response(
        {"stories": stories, "num_texts_in_stories": num_texts_in_stories}
    )


@app.route("/", methods=["GET"])
def hello_world():
    return "Linker-API is running!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
