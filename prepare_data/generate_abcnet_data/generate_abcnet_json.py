#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import sys
import cv2
from matplotlib.pyplot import box
import numpy as np
from shapely.geometry import *
import argparse
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", required=True)
    parser.add_argument("--output_data_folder", required=True)
    return parser.parse_args()

# Desktop Latin_embed.
cV2 = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

dataset = {
    'licenses': [],
    'info': {},
    'categories': [],
    'images': [],
    'annotations': []
}

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"

def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()
TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D^", "d^"]

CTLABELS = [
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
    "ˋ",
    "ˊ",
    "﹒",
    "ˀ",
    "˜",
    "ˇ",
    "ˆ",
    "˒",
    "‑",
]

def parse_tone(word):
    res = ""
    tone = ""
    for char in word:
        if char in dictionary:
            for group in groups:
                if char in group:
                    if tone == "":
                        tone = TONES[group.index(char)]
                    res += group[0]
        else:
            res += char
    res += tone
    return res


def full_parse(word):
    word = parse_tone(word)
    res = ""
    for char in word:
        if char in SOURCES:
            res += TARGETS[SOURCES.index(char)]
        else:
            res += char
    return res


def correct_tone_position(word):
    word = word[:-1]
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    replace_char = correct_tone_position(recognition)
    if recognition[-1] in TONES:
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition



def get_category_id(cls):
  for category in dataset['categories']:
    if category['name'] == cls:
      return category['id']

def generating_abcnetv2_json(input_data_folder, output_data_folder):
    images_path = input_data_folder + "images"
    root_path = ""
    phase = "train"
    _indexes_train = sorted([f
                    for f in os.listdir(os.path.join(root_path, './txt_abcnet_gen_train'))])
    _indexes = []
    _indexes.extend(_indexes_train)
    classes = ["text"]
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({
            'id': i,
            'name': cls,
            'supercategory': 'beverage',
            'keypoints': ['mean',
                        'xmin',
                        'x2',
                        'x3',
                        'xmax',
                        'ymin',
                        'y2',
                        'y3',
                        'ymax',
                        'cross']  # only for BDN
        })
    j = 1
    print("=== Start generating ABCnetV2 json ===")
    for enum,index in enumerate(tqdm(_indexes_train)):

        for ext in ['.png',',jpeg','.jpg']:
            exist = False
            path = os.path.join(root_path, images_path, os.path.splitext( str(index) )[0] + ext)
            if os.path.exists(path):
                im = cv2.imread(path)
                img_filename = os.path.basename(path)
                exist = True
                break
        if not exist:
            continue
        height, width, _ = im.shape
        dataset['images'].append({
            'coco_url': '',
            'date_captured': '',
            'file_name': img_filename,
            'flickr_url': '',
            'id': str(index.split(".")[0]),
            'license': 0,
            'width': width,
            'height': height
        })
        anno_file = os.path.join(root_path, './txt_abcnet_gen_train/') + index

        with open(anno_file, encoding="utf8") as f:
            lines = [line for line in f.readlines() if line.strip()]
            for i, line in enumerate(lines):
                pttt = line.strip().split('||||')
                parts = pttt[0].split(',')
                ct = pttt[-1].strip()

                cls = 'text'
                segs = [float(kkpart) for kkpart in parts[:16]]  
                
                xt = [segs[ikpart] for ikpart in range(0, len(segs), 2)]
                yt = [segs[ikpart] for ikpart in range(1, len(segs), 2)]
                xmin = min([xt[0],xt[3],xt[4],xt[7]])
                ymin = min([yt[0],yt[3],yt[4],yt[7]])
                xmax = max([xt[0],xt[3],xt[4],xt[7]])
                ymax = max([yt[0],yt[3],yt[4],yt[7]])
                box_width = max(0, xmax - xmin + 1)
                box_height = max(0, ymax - ymin + 1)
                if width == 0 or height == 0:
                    continue

                max_len = 100

                recs = [len(CTLABELS)+1 for ir in range(max_len)]

                ct =  str(ct)

                ct_parse = full_parse(ct)

                for ix, ict in enumerate(ct_parse):
                    if ix >= max_len: 
                        continue
                    if ict in CTLABELS:
                        recs[ix] = CTLABELS.index(ict)
                    else:
                        recs[ix] = len(CTLABELS)

                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [xmin, ymin, box_width, box_height],
                    'category_id': get_category_id(cls),
                    'id': j,
                    'image_id': str(index.split(".")[0]),
                    'iscrowd': 0,
                    'bezier_pts': segs,
                    'rec': recs
                })
                j += 1

    json_name = os.path.join(output_data_folder, '{}.json'.format(phase))
    with open(json_name, 'w') as f:
        json.dump(dataset, f, indent = 2, separators=(',', ':'))
    print("=== Done generating ABCnetV2 json ===")

if (__name__ == '__main__'):
    args = get_parser()
    generating_abcnetv2_json(args.input_data_folder, args.output_data_folder)
