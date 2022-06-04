import argparse
import glob
import os
import numpy as np
from PIL import Image, ExifTags, ImageOps
import random
from shapely.geometry import Polygon
from shutil import copy
from tqdm import tqdm

from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.warp import Curve, Distort, Stretch
from straug.weather import Fog, Snow, Frost, Rain, Shadow

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", required=True)
    parser.add_argument("--output_data_folder", required=True)
    parser.add_argument("--num_aug_imgs", type=int, required=True)
    return parser.parse_args()

def imreadRotate(img_filename):
    img = Image.open(img_filename)
    return ImageOps.exif_transpose(img)

def img_apply_filter(img):
    rng = np.random.default_rng(0)
    ops = [Shadow(rng), Contrast(rng), Brightness(rng), 
           Equalize(rng), Color(rng), GaussianNoise(rng)]
    
    num_filter = 1
    
    for i in range(num_filter):
        op = random.choice(ops)
        img = op(img, mag=3)
    return img

def augment_color(input_data_folder, output_data_folder, image_new_prefix, num_aug_imgs):
    start_pos = 0 # if needed, always -1

    files = sorted(glob.glob(input_data_folder + "images/*"))
    count_augmented_images = 0

    print("=== Start augmenting color ===")
    for i in tqdm(range(0, min(num_aug_imgs, len(files)))):
        file = files[i]
        file_ext = os.path.basename(file).split(".")[1]
        
        try:
            img = imreadRotate(file).convert('RGB')
            img = img_apply_filter(img)
            img.save(output_data_folder + "images/" + image_new_prefix + str(i + 1) + "." + file_ext)
            new_label_filename = output_data_folder + "labels/" + image_new_prefix + str(i + 1) + ".txt"
            copy(input_data_folder + "labels/" + os.path.basename(file).split(".")[0] + ".txt", new_label_filename)
            count_augmented_images += 1
        except Exception as e:
            print(e)

    print(f"=== Done augmenting color - {count_augmented_images} img(s) ===")


if (__name__ == "__main__"):
    args = get_parser()
    augment_color(args.input_data_folder, args.output_data_folder, "img_aug_color_", args.num_aug_imgs)


