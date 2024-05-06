from gensim.test.utils import common_texts
from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader as api
import json

def save_simple_model():
    model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")


def get_model(model_name=None):

    print(api.info())
    corpus = api.load("quora-duplicate-questions", return_path=True)

    corpus = api.load('text8')  # download the corpus and return it opened as an iterable
    model = Word2Vec(corpus)
    #model = KeyedVectors.load_word2vec_format(corpus, binary=False)
    #w = Word2Vec.load_word2vec_format(corpus, binary=True)
    return model




def get_keyed_vectors(words: list):

    # model = Word2Vec.load("word2vec.model")

    model = get_model()

    response = {}

    for word in words:
        if word in model.wv:
            response[word] = model.wv[word].tolist()

    json_response = json.dumps(response, indent=4)

    with open("test_json.json", "a+") as f:
        f.write(json_response)


    return response


words = [ "computer", "cmp", "comp", "television", "tv", "homework", "assignment", "ass", "home"] 
get_keyed_vectors(words)
