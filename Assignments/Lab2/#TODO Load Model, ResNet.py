#TODO Load Model, ResNet18,      DONE
#TODO Load and preprocess data, resize, normalize to ImageNet standards
#TODO Run inference, get predicted class and logits
#TODO Genereate CAM maps, torch-cam on the appropriate layer
#TODO Visualize results, overlay the CAM map on the original image
#TODO Analyze logits, look at top-k predictions for interesting cases

import json
import torch
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask

weights = get_model_weights("resnet18").DEFAULT
model = get_model("resnet18", weights=weights).eval()
preprocess = weights.transforms()

base_path = r"C:/Users/vilma/Machine-learning/Assignments/Lab2/Images/"

image_files = {
    "elephant": "img_african_elephant.jpg",
    "hippo": "img_hippo.jpg",
    "firetruck": "img_firetruck.jpg",
    "ambulance": "img_ambulance.jpg",
    "guitar": "img_acoustic_guitar.jpg",
    "violin": "img_violin.jpg"
}

images = {name: decode_image(base_path + file).float()/255.0 for name, file in image_files.items()}

inputs = {name: preprocess(img) for name, img in images.items()}

batch = torch.stack(list(inputs.values()))


cams = {}
with LayerCAM(model) as cam_extractor:
    outputs = {}
    for name, img in zip(inputs.keys(), batch):
        img = img.unsqueeze(0)
        output = model(img)
        class_idx = output.argmax(dim=1).item()
        cam = cam_extractor(class_idx, output)[0]
        cams[name] = cam
        outputs[name] = output

def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

for name, cam in cams.items():
    show_image(cam.squeeze(0).detach().numpy())


results = {}
for name in images:
    results[name] = overlay_mask(
        to_pil_image(images[name]),
        to_pil_image(cams[name].squeeze(0), mode='F'),
        alpha=0.5
    )

for res in results.values():
    show_image(res)


with open("C:\\Users\\vilma\\Machine-learning\\Assignments\\Lab2\\imagenet_class_index.json", 'r') as f:
    class_index = json.load(f)

def predict_class(probs, class_index):
    top_idx = int(probs.argmax())
    synset_id, class_name = class_index[str(top_idx)]

    return{
        "class_index": top_idx,
        "class_id": synset_id,
        "class_name": class_name,
        "confidence": float(probs[top_idx].item())
    }


predictions = []
for name, output in outputs.items():
    probs = output.softmax(dim=1).squeeze(0)
    predictions.append(predict_class(probs, class_index))

print(predictions)