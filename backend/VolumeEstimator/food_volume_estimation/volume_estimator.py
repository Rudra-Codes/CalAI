
import os
import numpy as np
import cv2
import json
import base64
# from scipy.spatial.distance import pdist
# from scipy.stats import skew
from keras.models import Model, model_from_json
import keras.backend as K
# from fuzzywuzzy import fuzz, process 
from PIL import Image
import io
from food_volume_estimation.depth_estimation.custom_modules import *
from food_volume_estimation.depth_estimation.project import *
from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
from food_volume_estimation.ellipse_detection.ellipse_detector import EllipseDetector
from food_volume_estimation.point_cloud_utils import *

BASE_DIR = os.path.dirname(__file__)

ARCHITECTURE_PATH = os.path.join(BASE_DIR, 'weights', 'architecture.json')  
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights', 'model_weights.h5')
SEGMENTATION_MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights', 'seg_weights.h5')

class VolumeEstimator():
    """Volume estimator object."""
    def __init__(self, depth_model_architecture, depth_model_weights,
        segmentation_model_weights):
        """Load depth model and create segmentator object."""
        with open(depth_model_architecture, 'r') as read_file:
            custom_losses = Losses()
            objs = {'ProjectionLayer': ProjectionLayer,
                    'ReflectionPadding2D': ReflectionPadding2D,
                    'InverseDepthNormalization': InverseDepthNormalization,
                    'AugmentationLayer': AugmentationLayer,
                    'compute_source_loss': custom_losses.compute_source_loss}
            model_architecture_json = json.load(read_file)
            self.monovideo = model_from_json(model_architecture_json,
                                              custom_objects=objs)
        self._VolumeEstimator__set_weights_trainable(self.monovideo,
                                                        False)
        self.monovideo.load_weights(depth_model_weights)
        self.model_input_shape = (
            self.monovideo.inputs[0].shape.as_list()[1:])
        depth_net = self.monovideo.get_layer('depth_net')
        self.depth_model = Model(inputs=depth_net.inputs,
                                    outputs=depth_net.outputs,
                                    name='depth_model')
        print('[*] Loaded depth estimation model.')

        # Depth model configuration
        MIN_DEPTH = 0.01
        MAX_DEPTH = 10
        self.min_disp = 1 / MAX_DEPTH
        self.max_disp = 1 / MIN_DEPTH
        self.gt_depth_scale = 0.35 # Ground truth expected median depth

        # Create segmentator object
        self.segmentator = FoodSegmentator(segmentation_model_weights)
        # Set plate adjustment relaxation parameter
        self.relax_param = 0.01
        

    def estimate_volume(self, input_image, fov=70,  plate_diameter_prior=0.3):
        """Volume estimation procedure.

        Inputs:
            input_image: Path to input image or image array.
            fov: Camera Field of View.
            plate_diameter_prior: Expected plate diameter.
            plot_results: Result plotting flag.
            plots_directory: Directory to save plots at or None.
        Returns:
            estimated_volume: Estimated volume.
        """
        # Load input image and resize to model input size
        if isinstance(input_image, str):
            if not os.path.isfile(input_image):
                raise FileNotFoundError(f"No such file or directory: '{input_image}'")
            orig_img = cv2.imread(input_image, cv2.IMREAD_COLOR)
            img = cv2.imread(input_image, cv2.IMREAD_COLOR)
        else:
            orig_img = input_image.copy()
            img = input_image
        input_image_shape = img.shape
        img = cv2.resize(img, (self.model_input_shape[1],
                               self.model_input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255

        # Create intrinsics matrix
        intrinsics_mat = self.__create_intrinsics_matrix(input_image_shape,
                                                         fov)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)

        # Predict depth
        img_batch = np.reshape(img, (1,) + img.shape)
        inverse_depth = self.depth_model.predict(img_batch)[0][0,:,:,0] 
        disparity_map = (self.min_disp + (self.max_disp - self.min_disp) 
                         * inverse_depth)
        depth = 1 / disparity_map
        # Convert depth map to point cloud
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        point_cloud_flat = np.reshape(
            point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))

        # Find ellipse parameterss (cx, cy, a, b, theta) that 
        # describe the plate contour
        ellipse_scale = 2
        ellipse_detector = EllipseDetector(
            (ellipse_scale * self.model_input_shape[0],
             ellipse_scale * self.model_input_shape[1]))
        ellipse_params = ellipse_detector.detect(input_image)
        ellipse_params_scaled = tuple(
            [x / ellipse_scale for x in ellipse_params[:-1]]
            + [ellipse_params[-1]])

        # Scale depth map
        if (any(x != 0 for x in ellipse_params_scaled) and
                plate_diameter_prior != 0):
            print('[*] Ellipse parameters:', ellipse_params_scaled)
            # Find the scaling factor to match prior 
            # and measured plate diameters
            plate_point_1 = [int(ellipse_params_scaled[2] 
                             * np.sin(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[1]), 
                             int(ellipse_params_scaled[2] 
                             * np.cos(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[0])]
            plate_point_2 = [int(-ellipse_params_scaled[2] 
                             * np.sin(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[1]),
                             int(-ellipse_params_scaled[2] 
                             * np.cos(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[0])]
            plate_point_1_3d = point_cloud[0, plate_point_1[0], 
                                           plate_point_1[1], :]
            plate_point_2_3d = point_cloud[0, plate_point_2[0], 
                                           plate_point_2[1], :]
            plate_diameter = np.linalg.norm(plate_point_1_3d 
                                            - plate_point_2_3d)
            scaling = plate_diameter_prior / plate_diameter
        else:
            # Use the median ground truth depth scaling when not using
            # the plate contour
            print('[*] No ellipse found. Scaling with expected median depth.')
            predicted_median_depth = np.median(1 / disparity_map)
            scaling = self.gt_depth_scale / predicted_median_depth
        depth = scaling * depth
        point_cloud = scaling * point_cloud
        point_cloud_flat = scaling * point_cloud_flat

        # Predict segmentation masks
        masks_array = self.segmentator.infer_masks(input_image)
        print('[*] Found {} food object(s) '
              'in image.'.format(masks_array.shape[-1]))

        # Iterate over all predicted masks and estimate volumes
        estimated_volumes = []
        for k in range(masks_array.shape[-1]):
            # Apply mask to create object image and depth map
            # --- 1. VOLUME ESTIMATION PREP (Downscaled) ---
            # Apply mask to create object depth map
            object_mask_2d = cv2.resize(masks_array[:,:,k], 
                                     (self.model_input_shape[1],
                                      self.model_input_shape[0]))
                                     
            # --- 2. VLM CROPPING (High-Resolution) ---
            # Use the original high-res mask for the bounding box
            orig_mask = masks_array[:, :, k]
            orig_y_indices, orig_x_indices = np.where(orig_mask > 0)
            
            if len(orig_y_indices) > 0 and len(orig_x_indices) > 0:
                y_min, y_max = np.min(orig_y_indices), np.max(orig_y_indices)
                x_min, x_max = np.min(orig_x_indices), np.max(orig_x_indices)
                
                # Optional: Add a 10px margin so the VLM has better visual context
                margin = 0
                y_min = max(0, y_min - margin)
                y_max = min(orig_img.shape[0], y_max + margin)
                x_min = max(0, x_min - margin)
                x_max = min(orig_img.shape[1], x_max + margin)
                
                high_res_crop = orig_img[y_min:y_max, x_min:x_max]
            else:
                high_res_crop = orig_img

            # Convert to base64
            # cv2.imencode returns a tuple (success_flag, buffer)
            success, buffer = cv2.imencode('.jpg', high_res_crop)
            if success:
                cropped_img = base64.b64encode(buffer).decode('utf-8') 
            else:
                cropped_img = None
                
            seg_img = Image.open(io.BytesIO(base64.b64decode(cropped_img)))
            seg_img.show()

            object_depth = object_mask_2d * depth
            # Get object points by filtering non-zero depth pixels
            object_mask_1d = (np.reshape(
                object_depth, (object_depth.shape[0] * object_depth.shape[1]))
                > 0)
            object_points = point_cloud_flat[object_mask_1d, :]

            # Filter outlier points
            object_points_filtered, sor_mask = sor_filter(
                object_points, 2, 0.7)
            # Estimate base plane parameters
            plane_params = pca_plane_estimation(object_points_filtered)
            # Transform object to match z-axis with plane normal
            translation, rotation_matrix = align_plane_with_axis(
                plane_params, np.array([0, 0, 1]))
            object_points_transformed = np.dot(
                object_points_filtered + translation, rotation_matrix.T)

            # Adjust object on base plane
            height_sorted_indices = np.argsort(object_points_transformed[:,2])
            adjustment_index = height_sorted_indices[
                int(object_points_transformed.shape[0] * self.relax_param)]
            object_points_transformed[:,2] += np.abs(
                object_points_transformed[adjustment_index, 2])
             
            # Estimate volume for points above the plane
            volume_points = object_points_transformed[
                object_points_transformed[:,2] > 0]
            estimated_volume, _ = pc_to_volume(volume_points)
            
            estimated_volumes.append({
                'segment_id': k+1,
                'volume_ml': estimated_volume*1e6,
                'image_base64': cropped_img
            })

        return estimated_volumes

    def __create_intrinsics_matrix(self, input_image_shape, fov):
        """Create intrinsics matrix from given camera fov.

        Inputs:
            input_image_shape: Original input image shape.
            fov: Camera Field of View (in deg).
        Returns:
            intrinsics_mat: Intrinsics matrix [3x3].
        """
        F = input_image_shape[1] / (2 * np.tan((fov / 2) * np.pi / 180))
        print('[*] Creating intrinsics matrix from given FOV:', fov)

        # Create intrinsics matrix
        x_scaling = int(self.model_input_shape[1]) / input_image_shape[1] 
        y_scaling = int(self.model_input_shape[0]) / input_image_shape[0] 
        intrinsics_mat = np.array(
            [[F * x_scaling, 0, (input_image_shape[1] / 2) * x_scaling], 
             [0, F * y_scaling, (input_image_shape[0] / 2) * y_scaling],
             [0, 0, 1]])
        return intrinsics_mat

    def __set_weights_trainable(self, model, trainable):
        """Sets model weights to trainable/non-trainable.

        Inputs:
            model: Model to set weights.
            trainable: Trainability flag.
        """
        for layer in model.layers:
            layer.trainable = trainable
            if isinstance(layer, Model):
                self.__set_weights_trainable(layer, trainable)
    

if __name__ == '__main__':
    # Note this was intended for testing purpose only.
    estimator = VolumeEstimator(ARCHITECTURE_PATH,
                          MODEL_WEIGHTS_PATH, 
                          SEGMENTATION_MODEL_WEIGHTS_PATH)
    image_path = BASE_DIR+"/uploads/test1.jpg"
    results = estimator.estimate_volume(image_path, fov=70)
    print([result["volume"] for result in results])
    # Iterate over input images to estimate volume
    


