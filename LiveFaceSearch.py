import cv2
import numpy as np

from modules import model_manager
from modules import display_tools

def draw_face_boxes(image, faces):
    # loop over the detected faces, mark the image where each face is found
    image_with_detections = image.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.imshow('main', image_with_detections)
    if len(faces) == 0:
        cv2.imshow('main', image)
#

def roi_from_model(pt_model, roi_shape, model_shape = (224,224)):
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
    pt_roi = np.array([0,0])

    pt_roi[0] = pt_model[0] * roi_shape[0] / model_shape[0]
    pt_roi[1] = pt_model[1] * roi_shape[1] / model_shape[1]
    return pt_roi

def camera_from_model(pt_model, roi_meta):
    roi_offset = np.array([roi_meta[0], roi_meta[1]])
    roi_shape = np.array([roi_meta[2], roi_meta[3]])
    pt = np.array([pt_model[0].item(), pt_model[1].item()])

    pt_camera = camera_from_roi(roi_from_model(pt, roi_shape), roi_offset)
    return pt_camera

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

def draw_glasses(image, roi, pts_model):
    """
    Display sunglasses on top of the image in the appropriate place
    """
    # pts_model.detach().cpu().numpy()
    key_pts = np.array([camera_from_model(pt, roi) for pt in pts_model])
    # roi_shape = roi[2:4]
    # pts_model = np.array([ [pt[0].item(), pt[1].item()] for pt in pts_model])
    # key_pts = np.array([ roi_from_model(pt, roi_shape) for pt in pts_model])

    # copy of the face image for overlay
    image_copy = np.copy(image)

    # top-left location for sunglasses to go
    # 17 = edge of left eyebrow
    x = int(key_pts[17, 0])
    y = int(key_pts[17, 1])

    # height and width of sunglasses
    h = int(abs(key_pts[27, 1] - key_pts[34, 1])) # h = length of nose
    w = int(abs(key_pts[17, 0] - key_pts[26, 0])) # w = left to right eyebrow edges


    # Shift/scale glasses image to match face
    w *=2
    h *= 2
    x -= w//4
    y -= h//2

    # read and resize sunglasses
    sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED) # read
    new_sunglasses = cv2.resize(sunglasses, (w, h), interpolation=cv2.INTER_CUBIC) #resize
    ind = np.argwhere(new_sunglasses[:, :, 3] > 0) # get indices of non-zero alpha channel

    # get region of interest on the face to change
    roi_color = image_copy[y:y + h, x:x + w]

    # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
    for i in range(3):
        roi_color[ind[:, 0], ind[:, 1], i] = new_sunglasses[ind[:, 0], ind[:, 1], i]
        # set the area of the image to the changed region with sunglasses
    image_copy[y:y + h, x:x + w] = roi_color

    cv2.imshow("glasses", image_copy)


def update_frame(image, model):
    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.2, 10)
    draw_face_boxes(image, faces)
    if len(faces) > 0:
        roi_meta = faces[0]
        roi = display_tools.DisplayTools.extract_roi(image, roi_meta)
        key_points = model.find_keypoints(roi) # Keypoints in model-space
        display_tools.DisplayTools.overlay_points(cv2.resize(roi, (224, 224)),
                                                  key_points,
                                                  name="ModelSpace")

        # roi_pts = np.array([pt for pt in key_points])
        roi_pts = np.array([roi_from_model(pt, roi.shape) for pt in key_points])
        display_tools.DisplayTools.overlay_points(roi,
                                                  roi_pts,
                                                  name="RoiSpace")
        draw_glasses(image, faces[0], key_points)


if __name__ == "__main__":
    vid = cv2.VideoCapture(0) # define a video capture object
    model = model_manager.ModelManager()
    while (True):
        ret, frame = vid.read() # Capture the video frame by frame
        update_frame(frame, model)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release() # After the loop release the cap object
    cv2.destroyAllWindows() # Destroy all the windows