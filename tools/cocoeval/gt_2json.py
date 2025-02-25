
import datetime
import glob
import json
import os
import cv2
import numpy as np

# 将yolo格式的数据集转换成coco格式的数据集
rootpath = '../txtoutputs/real/'
# 读取文件夹下的所有文件
images_path =rootpath# os.path.join(rootpath,"images")
labels_path = rootpath
output_path = './'
coco_json_save = output_path + '/gt_coco.json'

# 创建coco格式的json文件
coco_json = {
    'info': {
        "description": "mpj Dataset",
        "url": "www.mpj520.com",
        "version": "1.0",
        "year": 2022,
        "contributor": "mpj",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    },
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ],
    'images': [],
    'annotations': [],
    'categories': []
}

# 判断文件夹是否存在
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 读取classes.txt文件
classes = []
with open( './classes.txt', 'r') as f:
    classes = f.readlines()
    classes = [c.strip() for c in classes]

# 创建coco格式的json文件
for i, c in enumerate(classes):
    coco_json['categories'].append({'id': i + 1, 'name': c, 'supercategory': c})

allfiles = [x.split(os.sep)[-1].split(".")[0] for x in glob.glob(os.path.join(labels_path,"*.txt"))]
# 读取images文件夹下的所有文件
images = os.listdir(images_path)
for image in images:
    # 获取图片名和后缀
    image_name, image_suffix = os.path.splitext(image)

    if image_name not in allfiles:
        continue
    # 获取图片的宽和高
    image_path = images_path + '/' + image
    if image_path.endswith(".txt"):
        continue
    img = cv2.imread(image_path)
    print(image_path)
    height, width, _ = img.shape
    # 添加图片信息
    coco_json['images'].append({
        'id': image_name,
        'file_name': image,
        'width': width,
        'height': height,
        'date_captured': datetime.datetime.utcnow().isoformat(' '),
        'license': 1
    })
    # 读取图片对应的标签文件
    label_path = labels_path + '/' + image_name + '.txt'
    if not os.path.exists(label_path):
        continue
    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [l.strip() for l in labels]
        for j, label in enumerate(labels):
            label = label.split(' ')
            if len(label)<8:
                continue
            # 获取类别id
            category_id = int(label[0])
            points = np.array([float(x) for x in label[1:]]).reshape((-1,2))
            xmin = min(points[:,0])* width
            ymin = min(points[:,1])* height
            xmax = max(points[:,0])* width
            ymax = max(points[:,1])* height
            w = xmax-xmin
            h = ymax-ymin

            points[:,0] =  points[:,0]* width
            points[:,1] =  points[:,1]* height
            tt = []
            for p in points:
                tt.append(p[0])
                tt.append(p[1])
            coco_json['annotations'].append({
                'image_id': image_name,
                'category_id': category_id + 1,
                'bbox': [xmin, ymin, w, h],
                'id': len(coco_json['annotations']),
                'area': w * h,
                'iscrowd': 0,
                'segmentation': [tt],
                'attributes': ""
            })

# 保存json文件
with open(coco_json_save, 'w') as f:
    json.dump(coco_json, f, indent=2)

print(len(coco_json['images']), len(coco_json['annotations']), len(coco_json['categories']), 'Done!')

