import torch
import sys
import numpy as np
import cv2
import os
from pytorch_grad_cam import GradCAM
from torchvision import models, transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

sys.path.append("../segment-anything-main/")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def generate_masks(syn_imgs_ori, sam_path):
    '''
    the syn_imgs are imgs without preprocess
    '''

    all_masks = []

    sam = sam_model_registry["vit_h"](checkpoint=sam_path).cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Segmenting the images...")
    for img in tqdm(syn_imgs_ori):
        this_mask = []
        min_val = torch.min(img)
        max_val = torch.max(img)
        # normalize
        img = (img - min_val) / (max_val - min_val)
        img *= 255
        img = img.permute(1,2,0)
        img = img.numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        masks = mask_generator.generate(img)
        
        for _mask in masks:
            this_mask.append(torch.unsqueeze(torch.tensor(_mask['segmentation']), dim=0))
        this_mask = torch.cat(this_mask, dim=0).to("cpu")
        all_masks.append(this_mask)

    return all_masks


def generate_cams(syn_imgs_preprocess, syn_labels, expert_model, target_layer):
    '''
    the syn_imgs are imgs without preprocess
    '''
    gen_cams = []
    
    expert_model.eval()
    grad_cam = GradCAM(model=expert_model, target_layers=[target_layer])

    print("Generating CAMs using grad-CAM...")
    for img_prep, label in tqdm(zip(syn_imgs_preprocess, syn_labels)):

        input_tensor = img_prep.unsqueeze(0).requires_grad_(True)
        target_category = [ClassifierOutputTarget(label)]
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=target_category)
        # grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension
        gen_cams.append(torch.tensor(grayscale_cam))
    gen_cams = torch.cat(gen_cams, dim=0).to("cpu")
    return gen_cams


def select_masks(all_masks, gen_cams, sel_p=0.7):
    '''
    the syn_imgs are imgs without preprocess
    '''

    foreground_masks = []
    print("Selecting segmentations via grad-CAM...")
    for i in tqdm(range(len(all_masks))):
        this_masks = all_masks[i]
        this_weight_per_segment = -1 * torch.ones(len(this_masks))
        for j, mask in enumerate(this_masks):
            this_weight_per_segment[j] = torch.sum(mask * gen_cams[i]) / torch.sum(mask)


        slkt_segment_idxs = torch.argsort(this_weight_per_segment)[-int(sel_p * len(this_masks)):]

        this_foreground = torch.zeros_like(gen_cams[i])
        for idx in slkt_segment_idxs:
            this_foreground += this_masks[idx]
        
        foreground_masks.append(torch.unsqueeze(this_foreground, dim=0))

    foreground_masks = torch.cat(foreground_masks, dim=0).to("cpu")
    return foreground_masks



class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()














