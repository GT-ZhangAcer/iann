import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

labs_folder = r'train\datasets\train\masks'
imgs_folder = r'train\datasets\train\images'

lab_names = os.listdir(labs_folder)
for lab_name in tqdm(lab_names):
    lab_path = os.path.join(labs_folder, lab_name)
    img_path = os.path.join(imgs_folder, lab_name)
    lab = cv2.imdecode(np.fromfile(lab_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1), cv2.COLOR_BGR2RGB)
    if len(lab.shape) != 2:
        lab = lab[:, :, 0]
    tmp = np.zeros_like(lab)
    tmp[lab != 0] = 255
    lab = Image.fromarray(np.uint8(tmp))
    img = Image.fromarray(np.uint8(img))
    lab.save(lab_path.replace('masks', 'labs'))
    img.save(img_path.replace('images', 'imgs').replace('png', 'jpg'))
    # cv2不能保存中文路径
    # cv2.imwrite(img_path.replace('images', 'imgs').replace('png', 'jpg'), img)
    # cv2.imwrite(lab_path.replace('masks', 'labs'), lab)
    # break