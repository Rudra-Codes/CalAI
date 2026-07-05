import os
import numpy as np
import cv2
import tensorflow as tf
from food_volume_estimation.volume_estimator import VolumeEstimator
# from food_volume_estimation.depth_estimation.custom_modules import *
# from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
from flask import Flask, request, jsonify, abort
import base64

BASE_DIR = os.path.dirname(__file__)

ARCHITECTURE_PATH = os.path.join(BASE_DIR, 'food_volume_estimation', 'weights', 'architecture.json')  
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'food_volume_estimation', 'weights', 'model_weights.h5')
SEGMENTATION_MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'food_volume_estimation', 'weights', 'seg_weights.h5')

app = Flask(__name__)

# def load_volume_estimator(depth_model_architecture, depth_model_weights,
#         segmentation_model_weights):
#     """Loads volume estimator object and sets up its parameters."""
#     # Create estimator object and intialize
#     global estimator
#     estimator = VolumeEstimator(arg_init=False)
#     with open(depth_model_architecture, 'r') as read_file:
#         custom_losses = Losses()
#         objs = {'ProjectionLayer': ProjectionLayer,
#                 'ReflectionPadding2D': ReflectionPadding2D,
#                 'InverseDepthNormalization': InverseDepthNormalization,
#                 'AugmentationLayer': AugmentationLayer,
#                 'compute_source_loss': custom_losses.compute_source_loss}
#         model_architecture_json = json.load(read_file)
#         estimator.monovideo = model_from_json(model_architecture_json,
#                                               custom_objects=objs)
#     estimator._VolumeEstimator__set_weights_trainable(estimator.monovideo,
#                                                       False)
#     estimator.monovideo.load_weights(depth_model_weights)
#     estimator.model_input_shape = (
#         estimator.monovideo.inputs[0].shape.as_list()[1:])
#     depth_net = estimator.monovideo.get_layer('depth_net')
#     estimator.depth_model = Model(inputs=depth_net.inputs,
#                                   outputs=depth_net.outputs,
#                                   name='depth_model')
#     print('[*] Loaded depth estimation model.')

#     # Depth model configuration
#     MIN_DEPTH = 0.01
#     MAX_DEPTH = 10
#     estimator.min_disp = 1 / MAX_DEPTH
#     estimator.max_disp = 1 / MIN_DEPTH
#     estimator.gt_depth_scale = 0.35 # Ground truth expected median depth

#     # Create segmentator object
#     estimator.segmentator = FoodSegmentator(segmentation_model_weights)
#     # Set plate adjustment relaxation parameter
#     estimator.relax_param = 0.01

#     # Need to define default graph due to Flask multiprocessing
    
def load_models():
    global estimator
    estimator = VolumeEstimator(ARCHITECTURE_PATH,
                          MODEL_WEIGHTS_PATH, 
                          SEGMENTATION_MODEL_WEIGHTS_PATH)
    global graph
    graph = tf.get_default_graph()

@app.route('/predict', methods=['POST'])
def volume_estimation():

    img = None

    # Handle JSON requests
    if request.is_json:
        data = request.get_json()

        if not data or 'img' not in data:
            abort(400, "Missing 'img' field")

        try:
            image_b64 = data['img']

            # Support data:image/jpeg;base64,...
            if ',' in image_b64:
                image_b64 = image_b64.split(',', 1)[1]

            img_bytes = base64.b64decode(image_b64)
            np_img = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if img is None:
                abort(406, "Invalid image data")

        except Exception as e:
            abort(406, f"Failed to decode image: {str(e)}")

        try:
            plate_diameter = float(data.get('plate_diameter', 0.3))
        except (ValueError, TypeError):
            plate_diameter = 0.3

    # Handle multipart/form-data requests
    else:
        if 'img' not in request.files:
            abort(400, "No 'img' file provided in the request")

        try:
            file = request.files['img']
            img_bytes = file.read()
            np_img = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if img is None:
                abort(406, "Invalid image data")

        except Exception as e:
            abort(406, str(e))

        try:
            plate_diameter = float(request.form.get('plate_diameter', 0.3))
        except ValueError:
            plate_diameter = 0.3

    # Estimate volumes
    with graph.as_default():
        volumes = estimator.estimate_volume(
            img,
            fov=70,
            plate_diameter_prior=plate_diameter
        )

    return jsonify({
        'total_segments': len(volumes),
        'segments': volumes
    }), 200

load_models()
if __name__ == '__main__':
    
    
    app.run(host='0.0.0.0', port=8081)

