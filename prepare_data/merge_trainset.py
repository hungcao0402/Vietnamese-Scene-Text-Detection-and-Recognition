# import the necessary packages
import numpy as np
import cv2
import os
import re
from os import listdir
from os.path import isfile, join
import glob
import shutil
import argparse

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
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, default='/bkai/data',
                      help='path to root folder included bkai and vintext data')
                      
  parser.add_argument('--vintext', type=str, default='/bkai/data/vintext/vietnamese/',
                      help='path to root included vintext data')   
                             
  parser.add_argument('--bkai', type=str, default='/bkai/data/bkai',
                      help='path to root included bkai data') 
      
  parser.add_argument('--output_path', type=str, default='/bkai/data/trainset',
                      help='path to output merge 2k5 image data')       
  args = parser.parse_args()
                           
  root = args.root
  test_img = os.path.join(args.vintext, 'test_image/*')
  unseen_img = os.path.join(args.vintext, 'unseen_test_images/*')
  train_img = os.path.join(args.vintext,'train_images/*')
  label_vintext = os.path.join(args.vintext,'labels/*')
  bkai_img = os.path.join(args.bkai, 'training_img/*')
  bkai_label = os.path.join(args.bkai, 'training_gt/*')
  
  outpath = args.output_path
 
  img_vintext_paths = glob.glob(test_img) + glob.glob(unseen_img) + glob.glob(train_img)
  img_bkai_paths = glob.glob(bkai_img)
  if not os.path.exists(outpath):
    os.mkdir(outpath)
    os.mkdir(os.path.join(outpath,'images'))
    os.mkdir(os.path.join(outpath,'labels'))
  
 
  for path in img_vintext_paths:
      img_name = os.path.basename(path)
      img_name_new = img_name.replace('im', 'img_') #img_name_new example: img_0001.jpg   
      shutil.copy(path, os.path.join(outpath ,'images', img_name_new))
      
  for path in glob.glob(label_vintext):
      name_new = 'img_' +os.path.basename(path).split('.')[0][3:7].zfill(4) +'.txt'
      shutil.copy(path, os.path.join(outpath ,'labels', name_new))
  
  for path in img_bkai_paths:
      img_name = os.path.basename(path)
      img_name_new = 'img_' + str(int(img_name.split('.')[0].split('_')[1])+2000)  + '.jpg'
      shutil.copy(path, os.path.join(outpath ,'images', img_name_new))
      
  for path in glob.glob(bkai_label):
      name_new = 'img_' + str(int(os.path.basename(path).split('.')[0].split('_')[2]) + 2000) + '.txt'
      shutil.copy(path, os.path.join(outpath ,'labels', name_new))
      
  
if __name__ == "__main__":
    main()
  

