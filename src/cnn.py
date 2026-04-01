import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np


# Définition de l'architecture du réseau convolutionnel
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Premier bloc convolutionnel
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Deuxième bloc convolutionnel
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Troisième bloc convolutionnel
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couches entièrement connectées
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, dropout: bool = False):
        # Passage par les blocs convolutionnels
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # Aplatissement pour les couches entièrement connectées
        x = x.view(x.size(0), -1)

        # Passage par les couches entièrement connectées
        x = self.fc1(x)
        if dropout:
            x = self.dropout(x)
        x = self.fc2(x)

        return x


# Modèle CNN plus profond avec plus de couches et de filtres
class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCNN, self).__init__()
        # Premier bloc convolutionnel
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Deuxième bloc convolutionnel
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Troisième bloc convolutionnel
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couches entièrement connectées
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x, dropout: bool = False):
        # Premier bloc
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Deuxième bloc
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Troisième bloc
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Aplatissement et passage par les couches entièrement connectées
        x = x.view(x.size(0), -1)
        if dropout:
            x = self.dropout1(torch.relu(self.fc1(x)))
            x = self.dropout2(torch.relu(self.fc2(x)))
        else:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Fonction pour visualiser quelques images du dataset
def visualize_dataset(dataloader, classes):
    batch = next(iter(dataloader))
    images, labels = batch

    # Afficher quelques images du dataset
    plt.figure(figsize=(10, 6))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        plt.title(classes[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Fonction pour entraîner le modèle
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device="cuda"):
    since = time.time()

    # Pour stocker les statistiques d'entraînement
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Chaque epoch a une phase d'entraînement et une phase de validation
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Mode d'entraînement
            else:
                model.eval()  # Mode d'évaluation

            running_loss = 0.0
            running_corrects = 0

            # Itérer sur les données
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Remise à zéro des gradients du paramètre
                optimizer.zero_grad()

                # Propagation avant
                # Suivre l'historique seulement si on est en phase d'entraînement
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Rétropropagation + optimisation uniquement en phase d'entraînement
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistiques
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Enregistrer les statistiques
            if phase == "train":
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Copie du modèle si meilleure précision
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print(f"Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Meilleure précision sur validation: {best_acc:.4f}")

    # Charger les poids du meilleur modèle
    model.load_state_dict(best_model_wts)

    # Tracer les courbes d'entraînement
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Evolution de la perte")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train")
    plt.plot(val_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Evolution de la précision")

    plt.tight_layout()
    plt.show()

    return model


# Fonction pour tester le modèle
def test_model(model, testloader, device="cuda"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Précision sur l'ensemble de test: {accuracy:.2f}%")
    return accuracy


# Fonction pour visualiser quelques prédictions
def visualize_predictions(model, testloader, classes, device="cuda"):
    model.eval()

    # Obtenir un lot de données
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Prédictions
    images_device = images.to(device)
    with torch.no_grad():
        outputs = model(images_device)
        _, preds = torch.max(outputs, 1)

    # Afficher quelques images avec leurs étiquettes prédites
    plt.figure(figsize=(10, 6))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        plt.title(f"Pred: {classes[preds[i]]}\nReal: {classes[labels[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Programme principal
def main():
    # Vérifier si un GPU est disponible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de: {device}")

    # Transformations pour les données
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Chargement des données CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    valloader = DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Dictionnaire des chargeurs de données
    dataloaders = {"train": trainloader, "val": valloader}

    # Classes de CIFAR-10
    classes = (
        "avion",
        "automobile",
        "oiseau",
        "chat",
        "cerf",
        "chien",
        "grenouille",
        "cheval",
        "bateau",
        "camion",
    )

    # Visualiser quelques images du dataset
    visualize_dataset(trainloader, classes)

    # Créer le modèle CNN
    model = SimpleCNN(num_classes=10)
    # model = DeepCNN(num_classes=10)  # Utiliser le CNN plus profond
    model = model.to(device)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Planificateur de taux d'apprentissage
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Entraîner le modèle
    model = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=15, device=device
    )

    # Tester le modèle
    test_model(model, testloader, device=device)

    # Visualiser quelques prédictions
    visualize_predictions(model, testloader, classes, device=device)

    # Sauvegarder le modèle
    torch.save(model.state_dict(), "cnn_simple_cifar10.pth")
    print("Modèle sauvegardé avec succès !")


if __name__ == "__main__":
    main()
