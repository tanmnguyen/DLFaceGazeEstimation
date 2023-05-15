import cv2 
import torch 
import argparse 
import numpy as np
from utils.file import load_model
from utils.runtime import available_device

from torchvision import models, transforms


from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision import VisionExplainer
from omnixai.visualization.dashboard import Dashboard
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

from datasets.FaceDataset import FaceDataset

def main(args):
    # load pretrained model
    _type, model = load_model(args.model)
    model = model.to(available_device())

    # read image 
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Resize((224, 224)).transform(Image(img))
    
    img = Image(
        data=np.concatenate([
            img.to_numpy()
        ]),
        batched=True
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

    explainer = VisionExplainer(
        explainers=["gradcam"],
        mode="regression",
        model=model,                   
        preprocess=preprocess,        
        params={
            # Set the target layer for GradCAM
            "gradcam": {"target_layer": model.model.layer4[-1]},
        },
    )

    # Generate explanations 
    local_explanations = explainer.explain(img)

    # Launch the dashboard
    dashboard = Dashboard(
        instances=img,
        local_explanations=local_explanations,
    )

    dashboard.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="path to a pretrained model .pt file")

    parser.add_argument('-image',
                        '--image',
                        type=str,
                        required=True,
                        help="path to an image file")

    args = parser.parse_args()
    main(args)