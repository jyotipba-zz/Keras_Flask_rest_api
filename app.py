# Flask
from flask import Flask, jsonify, request

# utilities
import numpy as np
from PIL import Image
from io import BytesIO

# mask r cnn
from mrcnn.model import MaskRCNN
from keras.backend import clear_session

# helper function and class
from detection import PredictionConfig, make_prediction

def load_model():

    # model  configuration object
    cfg = PredictionConfig()
    global model
    # load model weight
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model.load_weights('../Mask_RCNN/mask_rcnn_only_crossings_0098.h5', by_name=True)
    # This is very important
    model.keras_model._make_predict_function()
    return model

# Declare a flask app
app = Flask(__name__)
model = None
model = load_model()


@app.route("/predict", methods=["POST"])
def predict():

    # initialize the data dictionary that will be returned from the  view
    data = {'success': False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        if request.files.get('image'):
            # read the image in PIL format
            image = request.files['image'].read()
            image = Image.open(BytesIO(image))

            # make prediction
            result = make_prediction(image, model)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for prob, rois in zip(result['scores'], result['rois']):
                r = {"confidence_score": float(prob), "Bbox": rois.tolist()}
                data["predictions"].append(r)

            # indicate that the request was a success
            data['success'] = True
        # return the data dictionary as a JSON response
    return jsonify(data)

if __name__ == '__main__':
    print("Loading Keras model and Flask starting server... please wait until server has fully started")
    app.run(host='0.0.0.0')
