from urllib import parse
from sanic import Sanic
from sanic.response import json
from requests import request
from numpy import argmax
from nltk import sent_tokenize

from core.api.api_wrapper import ClaimSpotterAPI

app = Sanic()
api = ClaimSpotterAPI()

def get_user_input(r, input_text, k="input_text"):
    try:
        if r.method == "GET":
            return parse.unquote_plus(input_text.strip())
        elif r.method == "POST":
            return r.json.get(k, "")
    except Exception as e:
        print(e)

    return ""

@app.route("/score/sentences/", methods=["POST"])
async def score_sentences(request):
    paragraphs = get_user_input(request, "", k="paragraphs")

    tokenized_paragraphs = [sent_tokenize(sentences) for sentences in paragraphs]
    scored_paragraphs = [api.batch_sentence_query(sentences) for sentences in tokenized_paragraphs]

    print(tokenized_paragraphs, scored_paragraphs)
    return json(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8009)