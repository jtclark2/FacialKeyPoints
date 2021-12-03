import cv2
import torch
import numpy as np
from models import Net
net = Net()

def find_keypoints(image, faces, model, display_face=True):
    pad = 50  # provide a buffer to frame the face

    image_copy = np.copy(image)
    # loop over the detected faces from your haar cascade
    for (x, y, w, h) in faces:
        # Select the face as a region of interest
        im_height, im_width, channels = image_copy.shape
        roi_orig = image_copy[max(0, y - pad):min(im_height, y + h + pad),
                   max(0, x - pad):min(im_width, x + w + pad)]

        roi = np.copy(roi_orig)

        # Convert the face region from RGB to grayscale
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # scale grayscale image from [0,255] to [0,1] (for the ml model)
        roi = roi / 255.

        # Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        roi = cv2.resize(roi, (224, 224))

        # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        if (len(roi.shape) == 2):
            roi_reshaped = roi.reshape(roi.shape[0], roi.shape[1], 1)

        roi_reshaped = roi_reshaped.transpose((2, 0, 1))  # initial image (RGB uses different order of dims than our pytorch model)
        roi_tensor = torch.from_numpy(roi_reshaped)  # convert to pytorch tensor
        roi_tensor = roi_tensor.type(torch.FloatTensor)  # convert type to FloatTensor
        roi_batch = roi_tensor.unsqueeze(0)  # add a dimension for the batch, making this a batch of 1

        output_pts = model(roi_batch)
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        predicted_pts = output_pts[0].data  # 0 is the batch index
        predicted_pts = predicted_pts * 50.0 + 100
        roi_disp = cv2.resize(roi_orig, (224, 224))
        if(display_face):
            for point in predicted_pts:
                pt = (int(point[0].item()), int(point[1].item()) )
                cv2.drawMarker(roi_disp, pt, (0, 0, 255), markerType=cv2.MARKER_STAR,
                               markerSize=3, thickness=2, line_type=cv2.LINE_AA)
            cv2.imshow("Face", roi_disp)

        return predicted_pts

def load_model():
    """load saved model parameters"""
    model_dir = 'saved_models/'
    model_name = 'final_model.pt'  #
    net.load_state_dict(torch.load(model_dir + model_name))
    return net

def draw_face_boxes(image, faces):
    # loop over the detected faces, mark the image where each face is found
    image_with_detections = image.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.imshow('main', image_with_detections)
    if len(faces) == 0:
        cv2.imshow('main', image)

def map_roi_to_main(pts, roi, win_size):
    model_size = (224, 224)
    roi_corner = roi[0:1]
    roi_size = roi[2:3]
    for pt in pts:
        win_size


def draw_glasses(image, roi, face, key_pts):
    """
    Display sunglasses on top of the image in the appropriate place
    """
    sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)

    # copy of the face image for overlay
    image_copy = np.copy(image)

    def map_roi_to_main(pts, roi):
        pass

    # top-left location for sunglasses to go
    # 17 = edge of left eyebrow
    x = int(key_pts[17, 0])
    y = int(key_pts[17, 1])

    # height and width of sunglasses
    # h = length of nose
    h = int(abs(key_pts[27, 1] - key_pts[34, 1]))
    # w = left to right eyebrow edges
    w = int(abs(key_pts[17, 0] - key_pts[26, 0]))

    # read in sunglasses
    sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
    # resize sunglasses
    new_sunglasses = cv2.resize(sunglasses, (w, h), interpolation=cv2.INTER_CUBIC)

    # get region of interest on the face to change
    roi_color = image_copy[y:y + h, x:x + w]

    # find all non-transparent pts
    ind = np.argwhere(new_sunglasses[:, :, 3] > 0)

    # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
    for i in range(3):
        roi_color[ind[:, 0], ind[:, 1], i] = new_sunglasses[ind[:, 0], ind[:, 1], i]
        # set the area of the image to the changed region with sunglasses
    image_copy[y:y + h, x:x + w] = roi_color

    # display the result!
    plt.imshow(image_copy)



def update_frame(image, model):
    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.2, 10)
    draw_face_boxes(image, faces)
    find_keypoints(image, faces, model)




if __name__ == "__main__":
    vid = cv2.VideoCapture(0) # define a video capture object
    model = load_model()
    while (True):
        ret, frame = vid.read() # Capture the video frame by frame
        update_frame(frame, model)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release() # After the loop release the cap object
    cv2.destroyAllWindows() # Destroy all the windows