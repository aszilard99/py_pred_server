from flask import Flask
from InferenceService import predict
app = Flask(__name__)


@app.route('/')
def hello_geek():
    res = predict()
    return f"<h2>Prediction result:</h2> <b>{res[0][0]}</b>"


if __name__ == "__main__":
    app.run(debug=True)
