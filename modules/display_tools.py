import cv2
import numpy as np
from . import converters

def overlay_points(image, pts, name):
    im_copy = np.copy(image)
    for point in pts:
        pt = (int(point[0].item()), int(point[1].item()))
        cv2.drawMarker(im_copy, pt, (0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=3, thickness=2, line_type=cv2.LINE_AA)
    cv2.imshow(name, im_copy)
    return im_copy


def extract_roi(image, roi_meta, pad=50):
    (x, y, w, h) = roi_meta
    im_height, im_width, channels = image.shape
    roi = image[max(0, y - pad):min(im_height, y + h + pad),
               max(0, x - pad):min(im_width, x + w + pad)]
    return roi


def draw_glasses(image, roi, pts_model):
    """
    Display sunglasses on top of the image in the appropriate place
    """
    key_pts = np.array([converters.camera_from_model(pt, roi) for pt in pts_model])

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

