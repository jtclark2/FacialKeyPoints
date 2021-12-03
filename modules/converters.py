"""
These tools convert between images of different sizes.
Model space is a fixed size that the model was trained for.
ROI space is dependent on whatever region is being manipulated.
Camera is a full image (not a subset of anything else.
pad is a buffer optionally applied to the edge of roi
"""


import numpy as np

def roi_from_model(pt_model, roi_shape, model_shape=(224, 224)):
    """
    Convert coordinates from model back to the roi it was sampled from.

    :param pt_model: (x,y) point in model coordinates
    :param roi: (x, y, w, h)
        x and y are coordinates of the roi in the image (not used here)
        w and h are width and height
    :param model_shape: tuple(x,y) representing the size of the model input
        For current model -> fixed 224x224
    :return: pt in roi coordinates
    """
    model_shape = np.array(model_shape)
    pt_roi = np.array([0, 0])

    pt_roi[0] = pt_model[0] * roi_shape[0] / model_shape[0]
    pt_roi[1] = pt_model[1] * roi_shape[1] / model_shape[1]
    return pt_roi

def camera_from_roi(pt_roi, roi_offset):
    """
    Converts points in the roi back to camera coodinates

    Assumptions: roi is always fully contains in camera image.

    :param pt_roi: (x,y) pt in roi coorindates (top-left of roi = (0,0)
    :param roi_offset: (x,y) offset to corner of roi
    :return: pt in camera coodinates
    """
    pt_camera = pt_roi + roi_offset
    return pt_camera

def camera_from_model(pt_model, roi_meta):
    """
    Covert to camera space from model space.
    Note that camera is not needed. Just where roi is within the camera image.
        Assumption: All points are within the camera image. It's a good
        assumption for this application, though I'd want to guard against it
        if this was built for a larger code base.

    :param pt_model: point in model space
    :param roi_meta: (x, y, w, h) meta data for the region of interest
    :return:
        pt in camera space
    """
    roi_offset = np.array([roi_meta[0], roi_meta[1]])
    roi_shape = np.array([roi_meta[2], roi_meta[3]])
    pt = np.array([pt_model[0].item(), pt_model[1].item()])

    pt_camera = camera_from_roi(roi_from_model(pt, roi_shape), roi_offset)
    return pt_camera

