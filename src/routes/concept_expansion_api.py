from datetime import datetime, timedelta
from flask import jsonify, abort, request, Blueprint
from main.extract_concepts import extract_concepts, expand_concepts
from main.bloom_verbs import extract_bloom_verbs, get_bloom_taxonomy
from main.fetch_lo import get_top_sentences
from main.predict_bloom_taxonomy import predict_bloom_taxonomy

from main.predict_taxonomy import predict_taxonomy

REQUEST_API = Blueprint('request_api', __name__)

def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API

@REQUEST_API.route('/concept/extract', methods=['POST'])
def concept():
    """triggers code to extract concepts
    """
    print("request*****", request.files)

    if not request.files:
        abort(400)
    body = request.files["document"]
    keyphrases = extract_concepts(body.read().decode("utf-8"))
    
    return jsonify({"keywords": keyphrases})

@REQUEST_API.route('/concept/expand', methods=['POST'])
def concept_expand():
    """triggers code to expand concepts
    """
    if not request.files:
        abort(400)
    body = request.files["document"]
    keyphrases = expand_concepts(body.read().decode("utf-8"))
    
    return jsonify({"keywords": keyphrases})



@REQUEST_API.route('/getbloomverbs/<string:skillname>', methods=['POST'])
def fetch_bloom_verbs(skillname):
    """triggers code to expand concepts
    """
    if not request.files:
        abort(400)
    body = request.files["document"]
    text = body.read().decode("utf-8")
    bloom_verbs = extract_bloom_verbs(text,skillname)

    return jsonify({"bloomverbs": bloom_verbs})



@REQUEST_API.route('/getcognitivetaxonomy', methods=['POST'])
def get_cognitive_taxonomy():
    """triggers code to expand concepts
    """
    if not request.files:
        abort(400)
    body = request.files["document"]
    text = body.read().decode("utf-8")
    bloom_verbs = get_bloom_taxonomy(text)

    return jsonify({"bloomtaxonomy": bloom_verbs})


@REQUEST_API.route('/getcognitivecomplexity/<string:difficulty_level>/<string:taxonomy>', methods=['POST'])
def get_cognitive_complexity(difficulty_level,taxonomy):
    """triggers code to predict bloom taxonomy
    """
    print(request)
    # if not request.files:
    #     abort(400)
    body = request.files["document"]
    print("taxonomy",taxonomy)
    text = body.read().decode("utf-8")
    bloom_verbs = predict_bloom_taxonomy(text,taxonomy,difficulty_level)

    return jsonify({"bloomtaxonomy": [bloom_verbs]})

@REQUEST_API.route('/predicttaxonomy', methods=['POST'])
def predict_tax():
    """triggers code to predict hierarchy
    """
    if not request.files:
        abort(400)
    body = request.files["document"]
    text = body.read().decode("utf-8")
    output = predict_taxonomy(text)

    return jsonify({"bloomtaxonomy": output})



@REQUEST_API.route('/fetchrankedlo', methods=['POST'])
def fetch_lo():
    """triggers code to fetch ranked los
    """
    if not request.files:
        abort(400)
    body = request.files["document"]
    text = body.read().decode("utf-8")
    output = get_top_sentences(text)

    return jsonify({"rankedlo": output})


MODEL_HEALTH = {
    "health": {
        'status': u'api is up',
        'timestamp': (datetime.today() - timedelta(1)).timestamp()
    },
}


@REQUEST_API.route('/health', methods=['GET'])
def get_health():
    """Return all book requests
    @return: 200: health of api\
    flask/response object with application/json mimetype.
    """
    return jsonify(MODEL_HEALTH)