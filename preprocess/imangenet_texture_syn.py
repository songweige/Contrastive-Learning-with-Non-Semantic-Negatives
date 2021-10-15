### script to generate ImageNet-texture dataset

import os
import cv2
import argparse
import numpy as np
from shutil import copyfile
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--imagenet_path', type=str, required=True)
parser.add_argument('--imagenet_t_path', type=str, required=True)
args = parser.parse_args()

cmd_template = './texture-synthesis --out-size 224 --out %s generate %s %s'

def generate_2patch(start, end):
    for class_name in class_names[start:end]:
        print(class_name)
        source_class_folder = os.path.join(source_sub_folder, class_name)
        target_class_folder = os.path.join(target_sub_folder, class_name)
        if not os.path.exists(target_class_folder):
            os.makedirs(target_class_folder)
        else:
            continue
        fns = os.listdir(source_class_folder)
        for fn in fns:
            target_file = os.path.join(target_class_folder, fn.replace('JPEG', 'png'))
            if os.path.exists(target_file)
            img = cv2.imread(os.path.join(source_class_folder, fn))
            h, w, _ = img.shape
            patch = img[h//2-48:h//2+48, w//2-48:w//2+48, :]
            texture1_file = os.path.join(target_class_folder, 'texture1_'+fn.replace('JPEG', 'png'))
            cv2.imwrite(texture1_file, patch)
            h_rand = int(np.random.random()*(h-96))+48
            w_rand = int(np.random.random()*(w-96))+48
            print(h_rand, w_rand, img.shape)
            patch = img[h_rand-48:h_rand+48, w_rand-48:w_rand+48, :]
            texture2_file = os.path.join(target_class_folder, 'texture2_'+fn.replace('JPEG', 'png'))
            cv2.imwrite(texture2_file, patch)
            os.system(cmd_template%(target_file, texture1_file, texture2_file))
            os.remove(texture1_file)
            os.remove(texture2_file)


source_folder = args.imagenet_path
target_folder = args.imagenet_t_path

os.makedirs(target_folder, exist_ok=True) 
for mode in ['train', 'val']:
    source_sub_folder = os.path.join(source_folder, mode)
    target_sub_folder = os.path.join(target_folder, mode)
    os.makedirs(target_sub_folder, exist_ok=True) 
    class_names = os.listdir(source_sub_folder)
    n_thread = 100
    n_classes = 1000//n_thread
    p = Pool(n_thread)
    for i in range(n_thread):
        p.apply_async(generate_2patch, args=((i+0)*n_classes, (i+1)*n_classes),)
    for class_name in class_names:
        print(class_name)
        source_class_folder = os.path.join(source_sub_folder, class_name)
        target_class_folder = os.path.join(target_sub_folder, class_name)
        if not os.path.exists(target_class_folder):
            os.makedirs(target_class_folder)
        fns = os.listdir(source_class_folder)
        for fn in fns:
            target_file = os.path.join(target_class_folder, fn.replace('JPEG', 'png'))
            if os.path.exists(target_file):
                continue
            img = cv2.imread(os.path.join(source_class_folder, fn))
            h, w, _ = img.shape
            patch = img[h//2-48:h//2+48, w//2-48:w//2+48, :]
            texture1_file = os.path.join(target_class_folder, 'texture1_'+fn.replace('JPEG', 'png'))
            cv2.imwrite(texture1_file, patch)
            h_rand = np.random.random()*(h-96)+48
            w_rand = np.random.random()*(w-96)+48
            patch = img[h_rand-48:h_rand+48, w_rand-48:w_rand+48, :]
            texture2_file = os.path.join(target_class_folder, 'texture2_'+fn.replace('JPEG', 'png'))
            cv2.imwrite(texture2_file, patch)
            os.system(cmd_template%(target_file, texture1_file, texture2_file))
            os.remove(texture1_file)
            os.remove(texture2_file)