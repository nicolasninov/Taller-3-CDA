import pandas as pd
import datetime as dt   
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import json
import logging
from waitress import serve

from typing import List

import settings
import routes
from data_model import DataModel
from prediction_models import PredictionModeloBaseline, PredictionModeloFinal
import random


app = Flask(__name__)
cors = CORS(app,resources={r'/api/*':{'origins':'*'}})


@app.route(routes.STATUS, methods=['GET'])
def health_check():
    return Response(response=json.dumps({
        'msg': 'Up and running',
        'version': settings.APP_VERSION,
        'deployed at': settings.DEPLOYED_AT,
        'requested at utc': dt.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S")
    }),
        headers={"Content-Type": "application/json"},
        status=200)

@app.route(routes.GET_PREDICTION_MODELO_BASELINE, methods=['POST'])
def make_predictions_modelo_baseline():
    X = request.json
    print(X)
    df = pd.DataFrame(X)
    print(df)
    predicion_model = PredictionModeloBaseline()
    results = predicion_model.make_predictions(df)
    return results

@app.route(routes.GET_PREDICTION_MODELO_FINAL, methods=['POST'])
def make_predictions_modelo_final():
    X = request.json
    print(X)
    df = pd.DataFrame(X)
    print(df)
    predicion_model = PredictionModeloFinal()
    results = predicion_model.make_predictions(df)
    return results

@app.route(routes.GET_EXPLAIN_MODELO_BASELINE, methods=['POST'])
def get_explanation_modelo_baseline():
    X = request.json
    print(X)
    df = pd.DataFrame(X)
    print(df)
    predicion_model = PredictionModeloBaseline()
    results = predicion_model.get_explanation()
    return results

@app.route(routes.GET_EXPLAIN_MODELO_FINAL, methods=['POST'])
def get_explanation_modelo_final():
    X = request.json
    print(X)
    df = pd.DataFrame(X)
    print(df)
    predicion_model = PredictionModeloFinal()
    results = predicion_model.get_explanation()
    return results

@app.route(routes.A_B_TEST_BASELINE, methods=['POST'])
def make_predictions_a_b_test_baseline():
    X = request.json
    print(X)
    df = pd.DataFrame(X)
    print(df)
    predicion_model = PredictionModeloBaseline()
    results = predicion_model.make_predictions_ab(df)
    print(results)
    return results

@app.route(routes.A_B_TEST_FINAL, methods=['POST'])
def make_predictions_a_b_test_final():
    X = request.json
    print(X)
    df = pd.DataFrame(X)
    print(df)
    predicion_model = PredictionModeloBaseline()
    results = predicion_model.make_predictions_ab(df)
    print(results)
    return results

#print(get_user_data('user3'))
if __name__ == "__main__":
    logger = logging.getLogger('waitress')
    logging.info('Server started.')
    serve(
        app=app,
        host='127.0.0.1',
        port=settings.PORT,
        threads=settings.WAITRESS_WORKERS,
        connection_limit=settings.WAITRESS_CHANNELS
    )