import argparse
import logging
import os
import glob
import tqdm
import torch
import PIL
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils import Config, Logger, CharsetMapper


def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model

def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    
    
    return (img-mean[...,None,None]) / std[...,None,None]
  

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == model_eval: output = res
        else: output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        
        return pt_text, pt_scores, pt_lengths
    
    output = _get_output(output, model_eval)
    
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)
    
    return pt_text, pt_scores, pt_lengths_

def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    print(file)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model

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
    parser.add_argument('--config', type=str, default='configs/train_iternet.yaml',
                        help='path to config file')
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str, default='~')
    parser.add_argument('--model_eval', type=str, default='alignment', 
                        choices=['alignment', 'vision', 'language'])
    parser.add_argument('--test_img', type=str, default='')
    args = parser.parse_args()
    config = Config(args.config)
    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint
    if args.model_eval is not None: config.model_eval = args.model_eval
    config.global_phase = 'test'
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    device = 'cpu' if args.cuda < 0 else f'cuda:{args.cuda}'
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    logging.info(config)
    
    logging.info('Construct model.')
    model = get_model(config).to(device)
    model = load(model, config.model_checkpoint, device=device)
    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)
    result_dir = 'submissions/'
    if os.path.isdir(args.input):
        paths = [os.path.join(args.input, fname) for fname in os.listdir(args.input)]
    else:
        paths = glob.glob(os.path.expanduser(args.input))
        assert paths, "The input path(s) was not found"
        
    paths = sorted(paths)
    
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, 'visualize'))
        os.makedirs(os.path.join(result_dir, 'predicted'))

   
    perspective = True
    vis_rec = True
    count=0
    result = []
    vis_conf = True
    count_drop = 0
    for detect in tqdm.tqdm(paths):
  
        vis_data = []
        path = os.path.join(args.input, detect)

        exts = ['.jpg', '.jpeg', '.png']
        im_name = ''
        for ext in exts:
            im_name = os.path.basename(detect.replace('.txt', ext).replace('res_',''))
            
            im_path = os.path.join(args.test_img, im_name)
      
            if os.path.exists(im_path):
                break
      
        if not os.path.exists(im_path):
            print(im_path)
            print('Img fail:', detect)
            continue
        im = cv2.imread(im_path)
     

        output = open(os.path.join(result_dir, 'predicted', 'res_'+im_name.replace('.jpg','') + '.txt'), 'w', encoding='utf-8')

        with open(path, 'r') as f:
            id = 0  
            lines = f.readlines()
           
            #lines = post_iou(lines)
            for bbox in lines:
                id += 1
                
                det_conf = 1.0
                if vis_conf:
                    det_conf = float(bbox.split(',',8)[-1])
                if det_conf < 0.1:
                  count_drop += 1
                  continue
                  
                bbox = list(map(float, bbox.strip().split(',')[:8]))
                xmin = int(min(bbox[:7:2]))
                xmax = int(max(bbox[:7:2]))
                ymin = int(min(bbox[1::2]))
                ymax = int(max(bbox[1::2]))
                
                if not perspective:
                    cropped = im[ymin:ymax, xmin:xmax]
                else:
                    cropped = perspective_transform(im, [(bbox[i], bbox[i+1]) for i in range(0, len(bbox) - 1, 2)])
                
                #cv2.imwrite(os.path.join('./vis', f'{os.path.basename(detect)[:-4]}_{id}.jpg'), cropped)     
                    
                img = preprocess(cropped, config.dataset_image_width, config.dataset_image_height)
                
                img = img.to(device)
                res = model(img)
                
                pt_text, rec_conf, __ = postprocess(res, charset, config.model_eval)
                
                pt_text=pt_text[0]
                rec_conf = rec_conf[0][0].item()

                if (rec_conf + det_conf) < 0.8 or rec_conf<0.3:
                 count_drop += 1
                 #print(os.path.basename(detect), pt_text + '\t' +str(round(det_conf,3)) + '\t' +str(round(rec_conf,3)) )
                 continue
                  
                count += 1
                
                data['result'].append({
                    'x_min': xmin,
                    'y_min': ymin,
                    'x_max': xmax,
                    'y_max': ymax,
                    'text': pt_text
                })


                points = list(zip(bbox[::2], bbox[1::2]))
                
                tmp = list(map(str, map(int, bbox)))
                tmp.append(pt_text)
                #logging.info(f'{os.path.basename(path)}_{id} : {pt_text} ')
                print(','.join(tmp), file=output)
                
        output.close()

        result.append(data)    
        

    print(f"Total box: {count}")
    print(f"Drop box: {count_drop}")
if __name__ == '__main__':
    main()
