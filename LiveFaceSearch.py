import cv2
import numpy as np

from modules import model_manager
from modules import display_tools
from modules import converters

def draw_face_boxes(image, faces):
    # loop over the detected faces, mark the image where each face is found
    image_with_detections = image.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.imshow('main', image_with_detections)
    if len(faces) == 0:
        cv2.imshow('main', image)
#


def update_frame(image, model):

    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.2, 10)

    draw_face_boxes(image, faces) # multiple faces can be detected

    if len(faces) > 0: # Let's play with the first face detection a bit more
        roi_meta = faces[0]
        roi = display_tools.extract_roi(image, roi_meta)
        key_points = model.find_keypoints(roi) # Keypoints in model-space
        display_tools.overlay_points(cv2.resize(roi, (224, 224)),
                                                  key_points,
                                                  name="ModelSpace")

        roi_pts = np.array([converters.roi_from_model(pt, roi.shape) for pt in key_points])
        display_tools.overlay_points(roi,
                                     roi_pts,
                                     name="RoiSpace")
        display_tools.draw_glasses(image, faces[0], key_points)


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