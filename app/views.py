from flask import Flask, jsonify, request

from image_classifier import base


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get the file from the request
        request_file = request.files['file']

        # convert to bytes
        img_bytes = request_file.read()

        class_id, class_name = base.predict(image_bytes=img_bytes)
        response = {'class_id': class_id, 'class_name': class_name}

        return jsonify(response)


if __name__ == '__main__':
    print('Take Home server started')
    app.run(debug=True, host='0.0.0.0')
