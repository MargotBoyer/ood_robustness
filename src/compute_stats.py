
import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import pandas as pd

from compute_ood_bounds import (ResNetTail, compute_bounds_tail_model, 
    nb_stable_actives, nb_stable_inactives, detect_ood, detect_ood_dataset,
    get_last_prerelu_layer, create_statistics_ood_dataset)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()


ARCH = "resnet18"
DATASET = "CIFAR10"
OOD_DATASET = "LSUN"  # example OOD dataset (same 3x32x32)
MODEL_PATH = f"/share/homes/boyerma/robustesse_ood/models/{ARCH}_{DATASET}.pth"


def get_model(arch, num_classes):
    if arch == "resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

num_classes = 10 if DATASET == "CIFAR10" else 100
model = get_model(ARCH, num_classes)
model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
for param in model.parameters():
    param.data = param.data.to(DEVICE)
model.eval()

# ----------------------
# OOD DATA
# ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Transform commun : resize + crop 32x32 + normalisation CIFAR10
transform_ood = transforms.Compose([
    transforms.Resize(36),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

if OOD_DATASET == "SVHN":
    oodset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
elif OOD_DATASET == "LSUN":
    # LSUN nécessite un téléchargement manuel (voir https://github.com/fyu/lsun)
    # classes possibles : 'bedroom', 'bridge', 'church_outdoor', 'classroom',
    #                     'conference_room', 'dining_room', 'kitchen',
    #                     'living_room', 'restaurant', 'tower'
    oodset = torchvision.datasets.LSUN(root='./data/lsun', classes=['bedroom_val'], transform=transform_ood)
elif OOD_DATASET == "DTD":
    # DTD (Describable Textures Dataset) — téléchargé automatiquement par torchvision
    oodset = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform_ood)
else:
    raise ValueError(f"OOD_DATASET '{OOD_DATASET}' non supporté. Choisir parmi : SVHN, LSUN, DTD.")

oodloader = torch.utils.data.DataLoader(oodset, batch_size=1, shuffle=True)
idloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
    batch_size=1, shuffle=True
)





CHOIX_LAYER_TAIL = 3
assert CHOIX_LAYER_TAIL in [0, 2,3, 4], "Invalid choice for tail layer. Must be 2 or 3."
EPSILON = 1e-5  # plus grand que input souvent
NORM = np.inf
METHOD = "alpha-CROWN"


tab_id = create_statistics_ood_dataset(
    full_model=model,
    dataloader=oodloader,
    EPSILON=EPSILON,
    DEVICE=DEVICE,
    NORM=NORM,      
    METHOD=METHOD,
    CHOIX_LAYER_TAIL=3,
    name=f"{OOD_DATASET}_little_epsilon"
)



