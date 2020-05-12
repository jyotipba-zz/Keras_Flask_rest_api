from mrcnn.config import Config
from keras.preprocessing.image import img_to_array


class PredictionConfig(Config):

    ''' configuration of model for inference '''

    NAME = "crossings_cfg20200105T1348"
    # Number of classes ( crossings + background )
    NUM_CLASSES = 1 + 1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 80
    DETECTION_MIN_CONFIDENCE = 0.80
     # setting Max ground truth insances
    MAX_GT_INSTANCES=5
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def make_prediction(img, model):

    img = img_to_array(img)
    img = img[:,:,:3]
    #model._make_predict_function()
    # detecting objects in the image
    prediction_result = model.detect([img])
    pred_result = prediction_result[0]
    return pred_result
