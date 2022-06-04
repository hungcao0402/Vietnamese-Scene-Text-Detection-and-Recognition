# import the necessary packages
import numpy as np
import cv2
import os
import re
from os import listdir
from os.path import isfile, join
import glob
import shutil

def order_points(pts):
    if isinstance(pts, list):
        pts = np.asarray(pts, dtype='float32')
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

    
def main():
  gt_path = '/bkai/data/trainset/labels/*'
  files = glob.glob(gt_path)
  rootImage = '/bkai/data/trainset/images'
  
  outpath = '/bkai/data/rec_train_data'
  # os.mkdir(outpath)
  if (not os.path.exists(outpath + '/img')):
    os.mkdir(outpath + '/img')

  crop_label = open(os.path.join(outpath,'crop_gt.txt'), 'w', encoding="utf-8")
  
  count = 0
  for f in files:
      fi = open(f,"r")
      lines = fi.readlines()
      fi.close()
      
      f = os.path.basename(f)
      img_name = f.split('.')[0] + '.jpg'
      img_path = os.path.join(rootImage, img_name)
      image = cv2.imread(img_path)
      
      id = 0
      for line in lines:
        line = line.strip()
        datas = line.split(",",8)
        bbox = list(map(int, datas[:8]))
        
        if datas[-1] == "###":
          continue
        
        pts = [(bbox[i], bbox[i+1]) for i in range(0, len(bbox) - 1, 2)]
        
        cropped_img = perspective_transform(image, pts)
        img_crop_name = img_name.split('.')[0] + '_' + str(id) + '.jpg'
        
        cv2.imwrite(os.path.join(outpath ,'img', img_crop_name), cropped_img)
        crop_label.write(f"img/{img_crop_name}\t{datas[-1]}\n")
     
        id += 1
        count += 1
      
  print('Total box:', count)
  
  
if __name__ == "__main__":
    main()
  

