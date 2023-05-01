import cv2 
import numpy as np

from utils.general import letterbox_resize
from GazeEstimation import GazeEstimationModel

class EyeGazeEstimationModel(GazeEstimationModel):
    def __init__(self, config_path: str):
        super(EyeGazeEstimationModel, self).__init__(config_path=config_path)
        # specify model name 
        self.model_name = "eye-" + self.model_name

    # override
    def _read_and_process(self, data_dict, indices): 
        images, labels = [], []
        for idx in indices:
            _images = data_dict[str(idx)]['images']
            _labels = data_dict[str(idx)]['labels']
            llmarks = data_dict[str(idx)]['left_landmarks']
            rlmarks = data_dict[str(idx)]['right_landmarks']
            # process 
            for i in range(len(_images)):
                l_eye = _images[i][
                    llmarks[i][1]: llmarks[i][3],
                    llmarks[i][0]: llmarks[i][2]
                ]
                r_eye = _images[i][
                    rlmarks[i][1]: rlmarks[i][3],
                    rlmarks[i][0]: rlmarks[i][2]
                ]
                # resize left eye image 
                l_eye = letterbox_resize(l_eye, (224, 112))
                # resize right eye image
                r_eye = letterbox_resize(r_eye, (224, 112))
                # stack image (mirror reverse)
                eye_img = np.hstack((r_eye, l_eye))
                # save eye image 
                images.append(eye_img)
                labels.append(_labels[i])

        # to numpy 
        images, labels = np.array(images), np.array(labels)
        # re-order shape
        images = images.transpose((0, 3, 1, 2))
        # result
        return images, labels