import cv2
import numpy as np

class DisplayTools:
    def __init__(self):
        pass

    @classmethod
    def overlay_points(self, image, pts, name):
        im_copy = np.copy(image)
        for point in pts:
            pt = (int(point[0].item()), int(point[1].item()))
            cv2.drawMarker(im_copy, pt, (0, 0, 255), markerType=cv2.MARKER_STAR,
                           markerSize=3, thickness=2, line_type=cv2.LINE_AA)
        cv2.imshow(name, im_copy)
        return im_copy

    @classmethod
    def extract_roi(self, image, roi_meta, pad=50):
        (x, y, w, h) = roi_meta
        im_height, im_width, channels = image.shape
        roi = image[max(0, y - pad):min(im_height, y + h + pad),
                   max(0, x - pad):min(im_width, x + w + pad)]
        return roi
