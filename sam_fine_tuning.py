# -*- coding: utf-8 -*-
"""
Fine-tuning du modèle SAM (Segment Anything Model) pour la segmentation d'images
Version simplifiée pour usage académique

Ce script permet de :
1. Charger et préparer des données d'images et de masques
2. Fine-tuner le modèle SAM sur un dataset personnalisé
3. Effectuer des prédictions avec le modèle entraîné

Basé sur le travail de Meta AI et adapté pour la segmentation d'objets spécifiques
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify
import random
from scipy import ndimage
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor, SamModel, SamConfig
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paramètres de patch pour diviser les grandes images
PATCH_SIZE = 256
STEP_SIZE = 256

# Paramètres d'entraînement
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1

# Chemins des fichiers (à adapter selon votre structure)
TRAINING_IMAGES_PATH = "training.tif"
TRAINING_MASKS_PATH = "training_mask.tif"
MODEL_SAVE_PATH = "mito_model_checkpoint.pth"

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_bounding_box(ground_truth_map):
    """
    Extrait les coordonnées de la boîte englobante à partir d'un masque

    Args:
        ground_truth_map: Masque binaire numpy array

    Returns:
        bbox: Liste [x_min, y_min, x_max, y_max] avec perturbation aléatoire
    """
    # Trouver les indices des pixels non-nuls
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Ajouter une perturbation aléatoire pour améliorer la robustesse
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return [x_min, y_min, x_max, y_max]

def load_and_prepare_data(images_path, masks_path):
    """
    Charge et prépare les données d'entraînement

    Args:
        images_path: Chemin vers le fichier d'images
        masks_path: Chemin vers le fichier de masques

    Returns:
        dataset: Dataset HuggingFace avec images et masques
    """
    print("Chargement des images et masques...")

    # Charger les grandes images et masques
    large_images = tifffile.imread(images_path)
    large_masks = tifffile.imread(masks_path)

    print(f"Images chargées: {large_images.shape}")
    print(f"Masques chargés: {large_masks.shape}")

    # Diviser en patches plus petits
    all_img_patches = []
    for img in range(large_images.shape[0]):
        large_image = large_images[img]
        patches_img = patchify(large_image, (PATCH_SIZE, PATCH_SIZE), step=STEP_SIZE)

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                all_img_patches.append(single_patch_img)

    images = np.array(all_img_patches)

    # Traiter les masques de la même manière
    all_mask_patches = []
    for img in range(large_masks.shape[0]):
        large_mask = large_masks[img]
        patches_mask = patchify(large_mask, (PATCH_SIZE, PATCH_SIZE), step=STEP_SIZE)

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i,j,:,:]
                single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
                all_mask_patches.append(single_patch_mask)

    masks = np.array(all_mask_patches)

    print(f"Patches d'images créés: {images.shape}")
    print(f"Patches de masques créés: {masks.shape}")

    # Supprimer les masques vides
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    filtered_images = images[valid_indices]
    filtered_masks = masks[valid_indices]

    print(f"Après filtrage - Images: {filtered_images.shape}, Masques: {filtered_masks.shape}")

    # Créer le dataset
    dataset_dict = {
        "image": [Image.fromarray(img) for img in filtered_images],
        "label": [Image.fromarray(mask) for mask in filtered_masks],
    }

    dataset = HFDataset.from_dict(dataset_dict)
    return dataset

def visualize_sample(dataset, index=None):
    """
    Visualise un échantillon du dataset

    Args:
        dataset: Dataset HuggingFace
        index: Index de l'échantillon à visualiser (aléatoire si None)
    """
    if index is None:
        index = random.randint(0, len(dataset)-1)

    example_image = dataset[index]["image"]
    example_mask = dataset[index]["label"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.array(example_image), cmap='gray')
    axes[0].set_title("Image")

    axes[1].imshow(example_mask, cmap='gray')
    axes[1].set_title("Masque")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# ============================================================================
# CLASSE DATASET PERSONNALISÉE
# ============================================================================

class SAMDataset(Dataset):
    """
    Dataset personnalisé pour l'entraînement de SAM
    Prépare les images et prompts pour le modèle
    """

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # Obtenir la boîte englobante comme prompt
        prompt = get_bounding_box(ground_truth_mask)

        # Préparer l'image et le prompt pour le modèle
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # Supprimer la dimension batch ajoutée par défaut
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Ajouter le masque de vérité terrain
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

# ============================================================================
# FONCTION D'ENTRAÎNEMENT
# ============================================================================

def train_model(train_dataloader, model, device):
    """
    Entraîne le modèle SAM

    Args:
        train_dataloader: DataLoader pour les données d'entraînement
        model: Modèle SAM à entraîner
        device: Device (CPU/GPU)
    """
    # Initialiser l'optimiseur et la fonction de perte
    optimizer = Adam(model.mask_decoder.parameters(), lr=LEARNING_RATE, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        print(f"Début de l'époque {epoch + 1}/{NUM_EPOCHS}")

        for batch in tqdm(train_dataloader, desc=f"Époque {epoch + 1}"):
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False
            )

            # Calculer la perte
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        print(f'Époque {epoch + 1} terminée')
        print(f'Perte moyenne: {mean(epoch_losses):.6f}')

    return model

# ============================================================================
# FONCTION D'INFÉRENCE
# ============================================================================

def predict_with_model(model, processor, image, device):
    """
    Effectue une prédiction avec le modèle entraîné

    Args:
        model: Modèle SAM entraîné
        processor: Processeur SAM
        image: Image PIL à segmenter
        device: Device (CPU/GPU)

    Returns:
        prediction: Masque de prédiction binaire
        probability: Carte de probabilité
    """
    model.eval()

    # Créer une grille de points comme prompt
    array_size = 256
    grid_size = 10

    x = np.linspace(0, array_size-1, grid_size)
    y = np.linspace(0, array_size-1, grid_size)
    xv, yv = np.meshgrid(x, y)

    xv_list = xv.tolist()
    yv_list = yv.tolist()
    input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)]
                   for x_row, y_row in zip(xv_list, yv_list)]
    input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)

    # Préparer les entrées
    inputs = processor(image, input_points=input_points, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # Appliquer sigmoid et convertir en masque binaire
    probability = torch.sigmoid(outputs.pred_masks.squeeze(1))
    probability = probability.cpu().numpy().squeeze()
    prediction = (probability > 0.5).astype(np.uint8)

    return prediction, probability

def visualize_prediction(image, prediction, probability):
    """
    Visualise les résultats de prédiction

    Args:
        image: Image originale
        prediction: Masque de prédiction
        probability: Carte de probabilité
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.array(image), cmap='gray')
    axes[0].set_title("Image originale")

    axes[1].imshow(probability)
    axes[1].set_title("Carte de probabilité")

    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title("Prédiction binaire")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale qui orchestre tout le processus
    """
    print("=== Fine-tuning SAM pour la segmentation d'images ===")

    # Vérifier la disponibilité du GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé: {device}")

    # 1. Charger et préparer les données
    print("\n1. Préparation des données...")
    dataset = load_and_prepare_data(TRAINING_IMAGES_PATH, TRAINING_MASKS_PATH)

    # Visualiser un échantillon
    print("Visualisation d'un échantillon du dataset:")
    visualize_sample(dataset)

    # 2. Initialiser le processeur et créer le dataset d'entraînement
    print("\n2. Initialisation du processeur SAM...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # 3. Charger le modèle
    print("\n3. Chargement du modèle SAM...")
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # Geler les encodeurs (seul le décodeur de masque sera entraîné)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    model.to(device)

    # 4. Entraîner le modèle
    print("\n4. Début de l'entraînement...")
    model = train_model(train_dataloader, model, device)

    # 5. Sauvegarder le modèle
    print(f"\n5. Sauvegarde du modèle dans {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 6. Test du modèle entraîné
    print("\n6. Test du modèle entraîné...")

    # Charger le modèle sauvegardé
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    trained_model = SamModel(config=model_config)
    trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    trained_model.to(device)

    # Tester sur un échantillon aléatoire
    test_idx = random.randint(0, len(dataset)-1)
    test_image = dataset[test_idx]["image"]

    prediction, probability = predict_with_model(trained_model, processor, test_image, device)
    visualize_prediction(test_image, prediction, probability)

    print("\n=== Entraînement terminé avec succès! ===")

if __name__ == "__main__":
    main()
