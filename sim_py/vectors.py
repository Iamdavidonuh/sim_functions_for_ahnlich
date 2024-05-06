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

    return response

sentences = [
    "Sunny weather is ideal for outdoor activities like playing football.",
    "Football fans enjoy gathering to watch matches at sports bars.",
    "On sunny days, people often gather outdoors for a friendly game of football.",
    "Grilling burgers and hot dogs is a popular activity during summer barbecues.",
    "Attending football games at the stadium is an exciting experience.",
    "Rainy weather can sometimes lead to canceled outdoor events like football matches."
]
get_keyed_vectors(sentences)
