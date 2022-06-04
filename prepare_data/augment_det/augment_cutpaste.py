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
    parser.add_argument("--input_cropped_bboxes_folder", required=True)
    parser.add_argument("--input_background_img_folder", required=True)
    parser.add_argument("--num_aug_imgs", type=int, required=True)
    parser.add_argument("--output_data_folder", required=True)
    return parser.parse_args()

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def get_bbox_pts(x1, y1, width, height):
    res = []
    res.append([x1, y1])
    res.append([x1 + width, y1])
    res.append([x1 + width, y1 + height])
    res.append([x1, y1 + height])
    return res

def scale_bbox(bb_image, bg_size):
    bg_width, bg_height = bg_size
    bb_width, bb_height = bb_image.size
    thresh = random.uniform(0.05, 0.1)
    min_width, min_height = int(thresh * bg_width), int(thresh * bg_height)
    bb_width = max(bb_width, min_width)
    bb_height = max(bb_height, min_height)
    bb_image = bb_image.resize((bb_width, bb_height))
    return bb_image

def get_random_bbox(bg_size, all_bb_imgs, all_bb_labels):
    bg_width, bg_height = bg_size
    while (True):
        bb_image_path = random.choice(all_bb_imgs)
        filename = os.path.basename(bb_image_path)
        for bb in all_bb_labels:
            bb_info = bb.split("\t")
            #print(bb_info)
            if (len(bb_info) != 2):
                continue
            if (filename == bb_info[0].replace("img/", "")):
                # join because some labels may contain space.
                full_text = bb_info[1]
                if ('#' not in full_text and len(full_text) > 1):
                    bb_image = Image.open(bb_image_path)
                    bb_image = scale_bbox(bb_image, bg_size=(bg_width, bg_height))
                    return (bb_image, filename, full_text) # full_text is the label of that bb

def augment_cutpaste(input_cropped_bboxes_folder, input_background_img_folder, output_data_folder, image_new_prefix, num_aug_imgs):
    # define
    images_dest = output_data_folder + "images/"
    labels_dest = output_data_folder + "labels/"

    # background image processing
    synthtext_bg = glob.glob(input_background_img_folder + "bg_img/*")

    f = open(input_background_img_folder + "im_names.txt", "r")
    lines = f.read().split("\n")
    f.close()

    bg_img_paths = []
    for line in lines:
        filepath = (input_background_img_folder + "bg_img/" + line)

        # check if exists
        if (filepath in synthtext_bg):
            bg_img_paths.append(filepath)

    # cropped bboxes processing
    all_bbox_images = glob.glob(input_cropped_bboxes_folder + "img/*")

    all_bbox_labels = []

    f = open(input_cropped_bboxes_folder + "crop_gt.txt", "r", encoding="utf-8")
    for label in f.read().split("\n"):
        if (label == ""):
            continue
        all_bbox_labels.append(label)
    f.close()

    # start
    print("=== Start augmenting cutpaste ===")

    for img_id in tqdm(range(num_aug_imgs)):
        num_bbox = random.randint(1, 10)

        bg_image = Image.open(random.choice(bg_img_paths))

        bg_width, bg_height = bg_image.size
        bbox_coords = []
        bbox_labels = []
        back_im = bg_image.copy()

        for i in range(num_bbox):
            bb_image, bb_label = None, None
            bb_width, bb_height = None, None
            x1, y1 = 0, 0
            cur_bb = []
            # check overlap
            while (True):
                bb_image, bb_filename, bb_label = get_random_bbox(bg_size=(bg_width, bg_height), all_bb_imgs=all_bbox_images, all_bb_labels=all_bbox_labels)
                bb_width, bb_height = bb_image.size
                x1, y1 = random.randint(0, bg_width), random.randint(0, bg_height) # top-left point
                cur_bb = get_bbox_pts(x1, y1, bb_width, bb_height)
                flag = True
                for pts in cur_bb:
                    if (pts[0] >= bg_width or pts[1] >= bg_height):
                        flag = False
                if (not flag):
                    continue
                for bb in bbox_coords:
                    iou = calculate_iou(cur_bb, bb)
                    if (iou > 0.0001):
                        flag = False
                if (flag):
                    break
            bbox_coords.append(cur_bb)
            bbox_labels.append(bb_label)
            back_im.paste(bb_image, (int(x1), int(y1)))
        
        new_image_name = image_new_prefix + str(img_id + 1)

        # save image
        back_im = back_im.convert('RGB')
        back_im.save(images_dest + new_image_name + ".jpg", quality=95)

        # save label
        content = ""
        for i in range(len(bbox_coords)):
            for pts in bbox_coords[i]:
                content += (str(pts[0]) + "," + str(pts[1]) + ",")
            content += (bbox_labels[i] + "\n")
        
        # write label file
        bb_label_file = open(labels_dest + new_image_name + ".txt", "w", encoding="utf-8")
        bb_label_file.write(content)
        bb_label_file.close()

    print(f"=== Done augmenting crop - {num_aug_imgs} img(s) ===")

if (__name__ == "__main__"):
    args = get_parser()
    augment_cutpaste(args.input_cropped_bboxes_folder, 
                    args.input_background_img_folder, 
                    args.output_data_folder, 
                    "img_aug_cutpaste_", 
                    args.num_aug_imgs)