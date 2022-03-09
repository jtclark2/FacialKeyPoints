import cv2
import numpy as np

from modules import model_manager
from modules import display_tools
from modules import converters


class ImageProcess:
    def __init__(self, avg=5):
        self.avg_duration = avg
        self.avg_pts = np.zeros((68, 2))

    def update_and_display_frame(self, image, model, face_cascade):
        # load in a haar cascade classifier for detecting frontal faces
        faces = face_cascade.detectMultiScale(image, 1.2, 10)


        if len(faces) > 0: # Let's play with the first face detection a bit more
            faces = sorted(faces, key=lambda x: x[2], reverse=True)
            # for face in faces:
            raw_face_data = faces[0]
            padded_face_data, cropped_face_im = display_tools.extract_roi(image, raw_face_data, pad=.09)
            if cropped_face_im is None:
                return
            key_points = model.find_keypoints(cropped_face_im) # Keypoints in model-space (224x224)

            crop_pts = np.array([converters.crop_from_model(pt, cropped_face_im.shape) for pt in key_points])

            model_space_image = cv2.resize(cropped_face_im, (224, 224))
            model_space_image = display_tools.overlay_points(model_space_image, key_points)
            cv2.imshow("ModelSpace", model_space_image)

            cropped_space_image = display_tools.overlay_points(cropped_face_im, crop_pts)
            cv2.imshow("CroppedSpace", cropped_space_image)

            image = display_tools.draw_glasses(image, padded_face_data, key_points, scale=1.25)

        with_glasses_image = display_tools.draw_face_boxes(image, faces)  # multiple faces can be detected
        cv2.imshow("With Shades", with_glasses_image)


if __name__ == "__main__":
    process = ImageProcess()
    vid = cv2.VideoCapture(0) # define a video capture object
    model = model_manager.ModelManager()
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

    while (True):
        ret, frame = vid.read() # Capture the video frame by frame
        frame = cv2.flip(frame, 1) # makes the cam mirror you - more intuitive
        process.update_and_display_frame(frame, model, face_cascade)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release() # After the loop release the cap object
    cv2.destroyAllWindows() # Destroy all the windows