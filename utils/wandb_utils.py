"""

Utility functions for wandb.

@author: Joshua Chough

"""

import wandb

# Generate interactive image mask from components
def wandb_mask(bg_img, pred_mask, true_mask, labels):
    return wandb.Image(bg_img, masks={
        "prediction" : {
            "mask_data" : pred_mask, 
            "class_labels" : labels
        },
        "ground truth" : {
            "mask_data" : true_mask, 
            "class_labels" : labels
        }
        }
    )