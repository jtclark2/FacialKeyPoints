import cv2
import numpy as np
import torch
from models import Net


class ModelManager:
    def __init__(self):
        """load saved model parameters"""
        model_dir = 'saved_models/'
        model_name = 'final_model.pt'  #
        self.model = Net()
        self.model.load_state_dict(torch.load(model_dir + model_name))

    def find_keypoints(self, roi_orig):
        """
        Find the facial keypoints in an image (expects the image to be a close-up of the face.

        :param roi_orig: The ROI, which should be a close-up of a face.
        :return:
            List of points in model space
        """

        # Pre-process image
        roi = np.copy(roi_orig)

        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) # Convert the face region from RGB to grayscale
        roi = roi / 255. # scale from [0,255] to [0,1]
        roi = cv2.resize(roi, (224, 224)) # Rescale the detected face to the CNN input (224x224)

        if (len(roi.shape) == 2): # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
            roi_reshaped = roi.reshape(roi.shape[0], roi.shape[1], 1)

        roi_reshaped = roi_reshaped.transpose((2, 0, 1))  # initial image (RGB uses different order of dims than our pytorch model)
        roi_tensor = torch.from_numpy(roi_reshaped)  # convert to pytorch tensor
        roi_tensor = roi_tensor.type(torch.FloatTensor)  # convert type to FloatTensor
        roi_batch = roi_tensor.unsqueeze(0)  # add a dimension for the batch, making this a batch of 1

        # run model to generate key points
        output_pts = self.model(roi_batch)
        output_pts = output_pts.view(output_pts.size()[0], 68, -1) # reshape as 68 pairs of points
        predicted_pts = output_pts[0].data  # drop the batch index (this is a single batch)
        predicted_pts = predicted_pts * 50.0 + 100

        return predicted_pts.numpy()
