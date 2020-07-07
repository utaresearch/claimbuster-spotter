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

@app.route("/score/batches/", methods=["POST"])
async def score_sentences(request):
    """
    Returns the scores of the batches of text provided.

    Parameters
    ----------
    paragraphs : List[string]
        Input batches to be scored.
    
    Returns
    -------
    Each sentence from each batch scored with the claimspotter model.

    <Response>
        Returns a response object with the body containing:
            List[List[Dict]]
            {
                `text` : string
                `index`: string
                `score`: float
                `result`: string
            }
    """
    paragraphs = get_user_input(request, "", k="paragraphs")

    tokenized_paragraphs = [sent_tokenize(sentences) for sentences in paragraphs]
    scored_paragraphs = [api.batch_sentence_query(sentences) for sentences in tokenized_paragraphs]

    results = [[{"text":tokenized_paragraphs[i][j], "index":j, "score":scored_paragraphs[i][j][1], "result":api.return_strings[argmax(scored_paragraphs[i][j])]} for j in range(len(tokenized_paragraphs[i]))] for i in range(len(tokenized_paragraphs))]
    return json(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8009)