import cv2
import numpy as np
from . import converters

def overlay_points(image, pts):
    im_copy = np.copy(image)
    for point in pts:
        pt = (int(point[0].item()), int(point[1].item()))
        cv2.drawMarker(im_copy, pt, (0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=1, thickness=2, line_type=cv2.LINE_AA)
    return im_copy


def extract_roi(image, face_data, pad):
    (x, y, w, h) = face_data
    pad = int( (w**2+h**2)**.5 *pad )
    im_height, im_width, channels = image.shape
    top = y-pad
    bottom = y+h+pad
    left = x-pad
    right = x + w + pad

    # reliably error free, but performance starts to suffer as face squishes on edges
    # roi = image[max(0, y - pad):min(im_height, y + h + pad),
    #            max(0, x - pad):min(im_width, x + w + pad)]

    if top < 0 or bottom > image.shape[0] or left < 0 or right > image.shape[1]:
        return None

    roi = image[top:bottom, left :right]
    padded_face_data = [x-pad, y-pad, w+2*pad, h+2*pad]
    return padded_face_data, roi

def draw_face_boxes(image, faces_data):
    # loop over the detected faces, mark the image where each face is found
    image_with_detections = image.copy()
    for (x,y,w,h) in faces_data:
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)
    if len(faces_data) == 0:
        return image
    else:
        return image_with_detections

    return

def draw_glasses(image, roi_data, pts_model, scale=1):
    """
    Display sunglasses on top of the image in the appropriate place
    """
    key_pts = np.array([converters.camera_from_model(pt, roi_data) for pt in pts_model])

    # copy of the face image for overlay
    image_copy = np.copy(image)

    # top-left location for sunglasses to go
    # 17 = edge of left eyebrow
    x = int(key_pts[17, 0])
    y = int(key_pts[17, 1])

    # height and width of sunglasses
    h = int(abs(key_pts[27, 1] - key_pts[34, 1]) * scale) # h = length of nose
    w = int(abs(key_pts[17, 0] - key_pts[26, 0]) * scale) # w = left to right eyebrow edges


    # Shift/scale glasses image to match face
    # arbitrary constant to shift glasses a bit, since image corners aren't perfectly aligned with eye corners (TODO: moves magic number)
    x = int(x + w//30)
    y = int(y - h//10)
    w = int(w)
    h = int(h)

    # Shift to adjust for scale factor
    x -= int((scale-1)*w/2)
    y -= int((scale-1)*h/2)

    if(w == 0 or h == 0):
        print("Warning: Image/glasses too small. Exiting without applying overlay.")
    print("scale: ", scale, w, h)
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

    return image_copy

