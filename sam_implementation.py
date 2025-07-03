import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import os

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('SAM - Segment Anything GUI')
        self.image_path = None
        self.image = None
        self.segmented_image = None
        self.mode = tk.StringVar(value='auto')
        self.points = []
        self.point_labels = []
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = None
        self.mask_generator = None
        self.setup_ui()

    def setup_ui(self):
        # Mode selection
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=5)
        tk.Label(mode_frame, text="Mode de segmentation :").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Automatique", variable=self.mode, value='auto').pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Prompt (points)", variable=self.mode, value='prompt').pack(side=tk.LEFT)

        # Image controls
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=5)
        tk.Button(ctrl_frame, text="Charger une image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Segmenter", command=self.segment_image).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Exporter résultat", command=self.export_result).pack(side=tk.LEFT, padx=5)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, width=512, height=512, bg='gray')
        self.canvas.pack(pady=10)
        self.canvas.bind('<Button-1>', self.on_canvas_click)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.jpeg *.png')])
        if not file_path:
            return
        self.image_path = file_path
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = img
        self.display_image(img)
        self.points = []
        self.point_labels = []
        self.segmented_image = None

    def display_image(self, img, points=None):
        img_disp = img.copy()
        if points:
            for (x, y, label) in points:
                color = (0,255,0) if label==1 else (255,0,0)
                cv2.circle(img_disp, (int(x), int(y)), 6, color, -1)
        img_pil = Image.fromarray(img_disp)
        img_pil = img_pil.resize((512, 512))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_canvas_click(self, event):
        if self.mode.get() != 'prompt' or self.image is None:
            return
        # Calculer la position dans l'image originale
        x = int(event.x * self.image.shape[1] / 512)
        y = int(event.y * self.image.shape[0] / 512)
        # Par défaut, label=1 (point positif). Shift+clic = négatif
        label = 0 if (event.state & 0x0001) else 1
        self.points.append((x, y, label))
        self.display_image(self.image, self.points)

    def segment_image(self):
        if self.image is None:
            messagebox.showerror('Erreur', 'Veuillez charger une image.')
            return
        if self.mode.get() == 'auto':
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            masks = self.mask_generator.generate(self.image)
            out_img = self.overlay_masks(self.image, masks)
            self.segmented_image = out_img
            self.display_image(out_img)
        else:
            if not self.points:
                messagebox.showerror('Erreur', 'Veuillez placer au moins un point sur l\'image.')
                return
            self.predictor = SamPredictor(self.sam)
            self.predictor.set_image(self.image)
            pts = np.array([[x, y] for (x, y, _) in self.points])
            labels = np.array([label for (_, _, label) in self.points])
            masks, scores, logits = self.predictor.predict(
                point_coords=pts,
                point_labels=labels,
                multimask_output=True
            )
            # On prend le meilleur masque
            idx = np.argmax(scores)
            mask = masks[idx]
            out_img = self.overlay_mask(self.image, mask)
            self.segmented_image = out_img
            self.display_image(out_img)

    def overlay_masks(self, image, masks):
        out = image.copy()
        for m in masks:
            mask = m['segmentation']
            color = np.random.randint(0,255,3)
            out[mask] = out[mask]*0.5 + color*0.5
        return out.astype(np.uint8)

    def overlay_mask(self, image, mask):
        out = image.copy()
        color = np.array([30, 144, 255])
        out[mask] = out[mask]*0.5 + color*0.5
        return out.astype(np.uint8)

    def export_result(self):
        if self.segmented_image is None:
            messagebox.showerror('Erreur', 'Aucun résultat à exporter.')
            return
        file_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not file_path:
            return
        img_bgr = cv2.cvtColor(self.segmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, img_bgr)
        messagebox.showinfo('Succès', f'Image exportée : {file_path}')

if __name__ == '__main__':
    root = tk.Tk()
    app = SAMGUI(root)
    root.mainloop()
