import torch
import cv2
import numpy as np
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from huggingface_hub import hf_hub_download
import albumentations as A
from albumentations.pytorch import ToTensorV2


REPO_ID = "JoshuaChick/TraversableGroundForRobot"
FILENAME = "TraversableGroundForRobot.pt"
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

IMAGE_PATH = "./image.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2",
    num_labels=1,
    ignore_mismatched_sizes=True
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

original_img = cv2.imread(IMAGE_PATH)
if original_img is None:
    print(f"Error: Could not find image at {IMAGE_PATH}")
    exit()

original_h, original_w = original_img.shape[:2]

# preprocess
image_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
input_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)

# inference
with torch.no_grad():
    outputs = model(input_tensor).logits

    upsampled_logits = nn.functional.interpolate(
        outputs,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    mask_prob = torch.sigmoid(upsampled_logits).squeeze().cpu().numpy()
    mask_binary = (mask_prob > 0.5).astype(np.uint8) * 255

final_mask = cv2.resize(mask_binary, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

# create the overlay
overlay = original_img.copy()
overlay[final_mask > 0] = [0, 255, 0] # BGR Green
result = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)

h, w = result.shape[:2]
max_dim = 900
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    result = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

cv2.imshow("SegFormer-B2 Traversable Ground Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()