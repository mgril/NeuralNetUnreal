import os
import shutil
from PIL import Image
import tkinter as tk
from tkinter import ttk

# Config 
SOURCE_FOLDER = "C:/Users/AutoRazer/Documents/GitHub/NeuralNetUnreal/Content/testSet/testSet"   
OUTPUT_FOLDER = "C:/Users/AutoRazer/Documents/GitHub/NeuralNetUnreal/Content/testSet/LabelizedImg"   

# Cree les dossiers 0-9
for i in range(10):
    os.makedirs(f"{OUTPUT_FOLDER}/{i}", exist_ok=True)

# Interface 
class Labelizer:
    def __init__(self):
        self.images = [f for f in os.listdir(SOURCE_FOLDER)
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.index  = 0
        self.skipped = 0

        self.root = tk.Tk()
        self.root.title("Labelizer — appuie sur 0-9 pour classer")
        self.root.geometry("400x500")
        self.root.configure(bg='black')

        # Compteur
        self.counter_label = tk.Label(
            self.root,
            text="",
            fg='white', bg='black',
            font=('Arial', 12)
        )
        self.counter_label.pack(pady=5)

        # Image
        self.image_label = tk.Label(self.root, bg='black')
        self.image_label.pack(expand=True)

        # Instructions
        tk.Label(
            self.root,
            text="Appuie sur 0-9 pour classer | S pour skip | Q pour quitter",
            fg='gray', bg='black',
            font=('Arial', 10)
        ).pack(pady=5)

        # Keybinds
        for digit in range(10):
            self.root.bind(str(digit), lambda e, d=digit: self.classify(d))
        self.root.bind('s', lambda e: self.skip())
        self.root.bind('q', lambda e: self.quit())

        self.show_image()
        self.root.mainloop()

    def show_image(self):
        if self.index >= len(self.images):
            self.quit()
            return

        # Compteur
        remaining = len(self.images) - self.index
        self.counter_label.config(
            text=f"Image {self.index + 1}/{len(self.images)} "
                 f"| Restantes: {remaining} "
                 f"| Skippées: {self.skipped}"
        )

        # Charge et affiche l'image
        img_path = f"{SOURCE_FOLDER}/{self.images[self.index]}"
        img = Image.open(img_path).convert('RGB')
        img = img.resize((300, 300), Image.NEAREST)  # Zoom pour bien voir

        from PIL import ImageTk
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image)

    def classify(self, digit):
        if self.index >= len(self.images):
            return

        src  = f"{SOURCE_FOLDER}/{self.images[self.index]}"
        dst  = f"{OUTPUT_FOLDER}/{digit}/{self.images[self.index]}"
        shutil.copy(src, dst)

        self.index += 1
        self.show_image()

    def skip(self):
        self.skipped += 1
        self.index += 1
        self.show_image()

    def quit(self):
        print(f"Termine ! {self.index} images traitees, {self.skipped} skippees")
        self.root.destroy()


Labelizer()