import os
from pprint import pprint

from flask import Flask, render_template, jsonify, request
from elasticsearch import Elasticsearch



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
