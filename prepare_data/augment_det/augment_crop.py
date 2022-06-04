import argparse
import glob
import os
import numpy as np
from PIL import Image, ExifTags, ImageOps
import random
from shapely.geometry import Polygon
from shutil import copy
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", required=True)
    parser.add_argument("--output_data_folder", required=True)
    parser.add_argument("--num_aug_imgs", type=int, required=True)
    return parser.parse_args()

def imreadRotate(img_filename):
    img = Image.open(img_filename)
    return ImageOps.exif_transpose(img)

def get_xy_minmax(p):
    # get xmin, ymin, xmax, ymax from polygon points
    xmin = min(p[0], p[2], p[4], p[6])
    ymin = min(p[1], p[3], p[5], p[7])
    xmax = max(p[0], p[2], p[4], p[6])
    ymax = max(p[1], p[3], p[5], p[7])
    return xmin, ymin, xmax, ymax

def get_freezone(pts, boundary):
    last_tail = 0
    current_state = 0
    results = []
    for p in pts:
        if (current_state == 0):
            results.append([last_tail, p[0]])

        if (p[1] == 0):
            # is first coord of that box
            current_state += 1
        else:
            current_state -= 1
            last_tail = p[0]
    
    if (last_tail <= boundary):
        results.append([last_tail, boundary])
        
    return results

def get_two_lines(freezone):
    mid = len(freezone) // 2
    first_line = random.choice(freezone[:mid])
    second_line = random.choice(freezone[mid:])
    
    first_line = random.randint(first_line[0], first_line[1])
    second_line = random.randint(second_line[0], second_line[1])
    
    return first_line, second_line

def augment_crop(input_data_folder, output_data_folder, image_new_prefix, num_aug_imgs):
    img_paths = sorted(glob.glob(input_data_folder + "images/*"))
    count_augmented_images = 0

    print("=== Start augmenting crop ===")

    for img_id in tqdm(range(min(num_aug_imgs, len(img_paths)))):
        img_path = img_paths[img_id]

        img = imreadRotate(img_path).convert('RGB')
        img_w, img_h = img.size
        
        # read bbox coordinates
        label_path = input_data_folder + "labels/" + os.path.basename(img_path).split(".")[0] + ".txt"

        with open(label_path, "r", encoding="utf-8") as f:
            rows = f.readlines()
        
        boxes = []
        
        for row in rows:
            box = row.replace("\n", "").split(",")
            if (box[-1] == "###"):
                continue
            for i in range(8):
                box[i] = max(0, int(box[i]))
            boxes.append(box)

        # declare a freezone
        x_freezone, y_freezone = [], []
        x_list, y_list = [], []
        
        for box in boxes:
            xmin, ymin, xmax, ymax = get_xy_minmax(box)
            x_list.append([xmin, 0])
            x_list.append([xmax, 1])
            y_list.append([ymin, 0])
            y_list.append([ymax, 1])
        x_list.sort()
        y_list.sort()

        x_freezone = get_freezone(x_list, boundary=img_w)
        y_freezone = get_freezone(y_list, boundary=img_h)
        
        counter = 0
        while (counter < 1000):
            try:
                x_first_line, x_second_line = get_two_lines(x_freezone)
                y_first_line, y_second_line = get_two_lines(y_freezone)
            except:
                print(img_path)
                print(boxes)
                print(x_list)
                print(y_list)
                print(x_freezone)
                print(y_freezone)
            
            new_img_w = x_second_line - x_first_line
            new_img_h = y_second_line - y_first_line
            
            if (new_img_w / new_img_h < 5 and new_img_h / new_img_w < 5):
                break
            counter += 1
        
        cropped_img = img.crop((x_first_line, y_first_line, x_second_line, y_second_line))
        
        label_file_content = ""
        for box in boxes:
            xmin, ymin, xmax, ymax = get_xy_minmax(box)
            if (    xmin >= x_first_line and xmin <= x_second_line
                and xmax >= x_first_line and xmax <= x_second_line
                and ymin >= y_first_line and ymin <= y_second_line
                and ymax >= y_first_line and ymax <= y_second_line):
                for i in range(8):
                    if (i % 2):
                        # y coord
                        box[i] -= y_first_line
                    else:
                        # x coord
                        box[i] -= x_first_line
                    label_file_content += (str(box[i]) + ",")
                label_file_content += (str(box[8]) + "\n")
                        
        if (label_file_content == "" or counter >= 1000):
            print("=== Empty file or cant find valid crop", img_path)
            continue
        
        # export to file 
        image_new_name = image_new_prefix + str(img_id + 1)
        cropped_img.save(output_data_folder + "images/" + image_new_name + ".jpg")
        
        with open(output_data_folder + "labels/" + image_new_name + ".txt", "w", encoding="utf-8") as f:
            f.write(label_file_content)
        
        count_augmented_images += 1
    
    print(f"=== Done augmenting crop - {count_augmented_images} img(s) ===")

if (__name__ == "__main__"):
    args = get_parser()
    augment_crop(args.input_data_folder, args.output_data_folder, "img_aug_crop_", args.num_aug_imgs)