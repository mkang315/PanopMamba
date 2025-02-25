import glob
import os

import cv2
import numpy as np

allfiles = []
with open("./MoNuSAC_images_and_annotations/mask/ImageSets/Segmentation/val.txt") as f:
    allfiles = [x.strip() for x in f.readlines()]
#注意修改了 local_visualizer.py中的 305-316
#                                141 188  还有 palette 颜色
# classes=( 'Epithelial', 'Lymphocyte', 'Neutrophil','Macrophage'),
palette=[  [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]

for file in glob.glob("./outputs_txt/predict/*.png"):
    name = file.split(os.sep)[-1].split(".png")[0]
    if name not in allfiles:
        continue
    img = cv2.imread(file )
    output_image = img.copy()
    h,w,c  = img.shape
    with open(os.path.join("./outputs_txt/predict",f"{name}.txt"),"w") as f:
        for cls in range(1,5):
            img = cv2.imread(file)

            img[img != cls] = 0
            img[img == cls] = 1
            img =  (img * 255).astype(np.uint8)
            cv2.imwrite("1.png",img)
            img = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
            img =  (img * 255).astype(np.uint8)
            # 找到轮廓
            contours,_= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            tmps = []
            for contour in contours:
                info = [cls-1]
                tmp = np.squeeze(contour)
                for t in tmp:
                    t = t.tolist()
                    if isinstance(t,int):
                        continue
                    info.append(t[0]/w)
                    info.append(t[1]/h)
                if len(info) <=1:
                    continue
                f.write(" ".join([str(x) for x in info]))
                f.write("\n")
            cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)  # 绘制所有轮廓
        cv2.imwrite(os.path.join("./outputs_txt/predict",f"{name}_.png"),output_image)
