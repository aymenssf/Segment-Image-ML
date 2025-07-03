# Segment Anything (SAM) - Exemples d'utilisation

## Prérequis

- Python 3.8+
- torch
- torchvision
- segment-anything (installé via GitHub)

## Installation

```bash
pip install -r requirements.txt
```

## Téléchargement du checkpoint

Téléchargez le fichier de checkpoint du modèle (ex: `sam_vit_h_4b8939.pth`) depuis [le dépôt officiel](https://github.com/facebookresearch/segment-anything#model-checkpoints) et notez son chemin.

## Recommandation

L'utilisation d'un GPU est fortement recommandée pour de meilleures performances.

## Utilisation

Modifiez les chemins dans `scenarios_sam.py` pour pointer vers votre checkpoint et votre image.

```python
sam_checkpoint = "chemin/vers/sam_vit_h_4b8939.pth"
image_path = "chemin/vers/image.jpg"
```

Lancez le script :

```bash
python scenarios_sam.py
```

Le script exécute deux scénarios :
- **Segmentation interactive** avec `SamPredictor`
- **Segmentation automatique** avec `SamAutomaticMaskGenerator`

Consultez le code pour des exemples de visualisation des résultats.
