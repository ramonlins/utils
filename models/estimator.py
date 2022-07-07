from typing import Tuple, List

import torch
from models.vgg_reg import VGGReg

class Estimator:
    def __init__(self, patch_res: Tuple[int, int], weights_path: str, model_name: str = 'vgg16'):
        """
        Loading model to make inference of patches
        
        Args:
            
            patch_res (Tuple): input image resolution (W x H)
            weights_path (string): path to load the model downloaded from azure
        """
        try:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            self.model = VGGReg(patch_res)
            
            self.model.load_state_dict(torch.load(weights_path))

            self.model.eval()

            self.model.to(device)

        except Exception as error:
            print(error)
            print("A problem has occurred while loading the model")

    def estimate(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Make inference

        Args:
            img (Tensor): Patch of image (N, W, H, 3)

        Returns:
            output (list): Density of each image (N, 1)
        """

        with torch.no_grad():
            output = self.model(img)

        return output