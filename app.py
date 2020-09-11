from urllib import parse
from sanic import Sanic
from sanic.response import json
from requests import request
from numpy import argmax
from nltk import sent_tokenize

from core.api.api_wrapper import ClaimSpotterAPI
from core.api.api_wrapper import FLAGS

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

def get_url_text(url):
    """
    Returns each sentence at the provided URL.

    Parameters
    ----------
    url : string
        Path to web page to score.

    Returns
    -------
    url_text: list[str] 
        List of each sentence in the page loaded at the given URL.
    """
    
    url_text = requests.request('GET', url).text
    url_text = url_text.replace("\r", "")

    
    return url_text

@app.route("/score/text/<input_text:(?!custom/).*>", methods=["POST", "GET"])
async def score_text(request, input_text):
    """
    Returns the scores of the text provided.

    Parameters
    ----------
    input_text : string
        Input text to be scored.
    
    Returns
    -------
    <Response>
        Returns a response object with the body containing a json-encoded dictionary containing the `claim`, it's `result`, and the `scores` associated with it.

        `claim` : string
        `results` : dict
            {
                `text` : string
                `index`: string
                `score`: float
                `result`: string
            }
    """
    input_text = get_user_input(request, input_text)
    sentences = sent_tokenize(input_text)
    scores = api.batch_sentence_query(sentences)

    results = [{"text":sentences[i], "index":i, "score":scores[i][1], "result":api.return_strings[argmax(scores[i])]} for i in range(len(sentences))]

    return json({'claim':input_text, 'results':results})


@app.route("/score/text-custom/<input_text:(?!custom/).*>", methods=["POST", "GET"])
async def score_text_custom_activation(request, input_text):
    """
    Returns the scores of the text provided.

    Parameters
    ----------
    input_text : string
        Input text to be scored.
    
    Returns
    -------
    <Response>
        Returns a response object with the body containing a json-encoded dictionary containing the `claim`, it's `result`, and the `scores` associated with it.

        `claim` : string
        `results` : dict
            {
                `text` : string
                `index`: string
                `score`: float
                `result`: string
            }
    """
    
    input_text = get_user_input(request, input_text)
    sentences = sent_tokenize(input_text)

    FLAGS.cs_custom_activation = True
    scores = api.batch_sentence_query(sentences)
    FLAGS.cs_custom_activation = False

    results = [{"text":sentences[i], "index":i, "score":scores[i][1], "result":api.return_strings[argmax(scores[i])]} for i in range(len(sentences))]

    return json({'claim':input_text, 'results':results})



@app.route("/score/url/<url:path>", methods=["POST", "GET"])
async def score_url(request, url):
    """
    Returns the scores of the text from the URL provided.

    Parameters
    ----------
    url : string
        Web page to be scored.
    
    Returns
    -------
    <Response>
        Returns a response object with the body containing a json-encoded list of dictionaries containing the `claim`, it's `result`, 
        and the `scores` associated with each claim on the web page.

        `claim` : string
        `results` : list[dict]
            {
                `text` : string
                `index`: string
                `score`: float
                `result`: string
            }
    """
    url = get_user_input(request, url, "url")
    url_text = get_url_text(url)

    sentences = sent_tokenize(url_text)

    scores = api.batch_sentence_query(sentences)

    results = [{"text":sentences[i], "index":i, "score":scores[i][1], "result":api.return_strings[argmax(scores[i])]} for i in range(len(sentences))]
    

    return json({'claim':input_text, 'results':results})

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

@app.route("/score/batches-custom/", methods=["POST"])
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
    
    FLAGS.cs_custom_activation = True
    scored_paragraphs = [api.batch_sentence_query(sentences) for sentences in tokenized_paragraphs]
    FLAGS.cs_custom_activation = False

    results = [[{"text":tokenized_paragraphs[i][j], "index":j, "score":scored_paragraphs[i][j][1], "result":api.return_strings[argmax(scored_paragraphs[i][j])]} for j in range(len(tokenized_paragraphs[i]))] for i in range(len(tokenized_paragraphs))]
    return json(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)