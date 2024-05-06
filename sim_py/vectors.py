from sentence_transformers import SentenceTransformer
import json

def get_model(model_name=None):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model



def get_keyed_vectors(sentences: list):

    model = get_model()
    response = {}
    for sentence in sentences:

        response[sentence] = model.encode(sentence).tolist()

    json_response = json.dumps(response, indent=4)

    with open("test_json.json", "a+") as f:
        f.write(json_response)

    print(response)

    return response


words = [ "computer", "cmp", "comp", "television", "tv", "homework", "assignment", "ass", "home"] 
get_keyed_vectors(words)
