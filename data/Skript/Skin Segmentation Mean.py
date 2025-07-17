import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


time = 60
fps = 30


def plot_skin_pixels(frame_rgb, mask):
    # Kopiere das Originalbild
    highlighted_image = frame_rgb.copy()
    height, width, _ = frame_rgb.shape

    # Erstelle einen grünen Hintergrund
    green_background = np.full_like(frame_rgb, [0, 0, 0])  # RGB für Grün [0, 255, 0]

    # Kombiniere das Bild mit der Hautmaske: Hautpixel bleiben wie sie sind, Nicht-Hautpixel werden grün

    plt.figure(figsize=(10, 5))

    # Originalbild anzeigen
    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    plt.title('Original RGB Bild')

    # Bild mit hervorgehobenen Hautpixeln anzeigen
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Erkannte Hautpixel hervorgehoben')

    plt.show()


def binary_img(image):  # Transformation eines Bildes in ein Binärbild mitels Otsu
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Transformation in Grauwertbild
    _, otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu Methode für Binärbild

    return otsu


def rem_line_contour(bin_image, original_image):  # Entfernen von Konturflächen mit bestimmter Größe
    original_image_2 = original_image.copy()  # Kopie erstellen, damit originales BIld nicht verändert wird

    # äußere Konturen finden, dabei möglichst wenige Punke verwenden
    contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:  # alle möglichen Konturen berücksichtigen
        mask = np.zeros_like(bin_image)  # schwarze Maske mit Größe von bin_image erstellen
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Konturen in Maske zeichnen und mit Weiß füllen

        pixel_count = np.sum(mask == 255)  # Weiße Pixel innerhalb jeder Kontur zählen

        if 17000 <= pixel_count <= 27000:  # Nur Konturen zwischen 17000 und 27000 Pixel in das Originale Bild zeichnen
            # schwarz füllen von Konturen, die im Bereich liegen
            cv2.drawContours(original_image_2, [contour], -1, (0, 0, 0), -1)

    return original_image_2


def filt_bin(image):  # Filterung + Binärbilderstellung

    flt = cv2.bilateralFilter(image, 50, 75, 75)  # Bilateraren Filter anwenden
    otsu = binary_img(flt)  # Obige Funktion zum erstellen von Binärbild mittels Otsu

    return otsu, flt


def process_image(bin_img):
    mask = np.zeros_like(bin_img)  # schwarze Maske der Größe von bin_img erstellen
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Konturen finden
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # Übrige Konturen weiß füllen

    return mask


def create_skin_mask(img):
    bin_img = binary_img(img)  # Binärbild erstellen
    mask1 = rem_line_contour(bin_img, img)  # Konturen erkennen und Pixelanzahl-Schwellwerte anwenden
    bin_filt, _ = filt_bin(mask1)  # Binalteralen Filer auf Maske anwenden und erneut Binärbild erstellen
    mask2 = process_image(bin_filt)  # Konturen von neuem Binärbild finden

    return mask2


def process_video_frames(folder_path, down_scaling_factor, window_length):
    files = [file for file in os.listdir(folder_path) if file.endswith('.bmp')]
    os.makedirs(folder_path + f'/csv/')

    df = pd.DataFrame(columns=["R", "G", "B"])

    for i in range(1, len(files) + 1):
        frame = cv2.imread(folder_path + "/image{}.bmp".format(i))

        # Konvertiere den Frame in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width = frame_rgb.shape[:2]

        # Nur bei erstem Durchlauf eine Maske generieren (Annahme: keine Bewegung der Hände)
        if i == 1:  # or (i % window_length) == 1:

            mask = create_skin_mask(frame_rgb)

        frame_rgb_ds = cv2.resize(frame_rgb, (width // down_scaling_factor, height // down_scaling_factor), interpolation=cv2.INTER_AREA)
        mask_ds = cv2.resize(mask, (width // down_scaling_factor, height // down_scaling_factor), interpolation=cv2.INTER_AREA)
        _, mask_ds = cv2.threshold(mask_ds, 1, 255, cv2.THRESH_BINARY)
        bool_mask = mask_ds.astype(bool)

        skin_pixels = frame_rgb_ds[bool_mask]
        skin_pixels_r = skin_pixels[:, 0].mean()
        skin_pixels_g = skin_pixels[:, 1].mean()
        skin_pixels_b = skin_pixels[:, 2].mean()

        # Plot Skin Mask
        #plot_skin_pixels(frame_rgb, mask_ds)

        # Finde die Indizes der True-Werte
        #y_indices, x_indices = np.nonzero(mask_ds)

        df.loc[len(df)] = [skin_pixels_r, skin_pixels_g, skin_pixels_b]

        print("Frame {} processed".format(i))

    df.to_csv(folder_path + "/csv/rgb_pixel_values_mean.csv", index=False)

pfad = ['32/A', '32/B', '32/C']

for messungen in pfad:
    #proband = '1'
    #  measurement = 'A'       # [D: Basline, E: Full Stenosis, F: Medium Stenosis]
    folder_path = f'F:/Probandenmessung/{messungen}'
    #folder_path = f'E:/Kaldenhoff/Probandenmessung/{proband}/{measurement}'
    #folder_path = f'{messungen}'
    print(folder_path)

    process_video_frames(folder_path, down_scaling_factor=8, window_length=300)
