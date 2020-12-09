import cv2
import os
import dlib
import sys

# 定义数据文件夹
out_dir = 'other_faces'
# 定义待提取数据文件夹，在这里用的是LFW的数据集
input_dir = 'input_img'
# 建立输出文件夹out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# 提取人脸的特征，使用dlib自带的检测器
size = 64
detector = dlib.get_frontal_face_detector()
# 读入输入文件夹中的图像，并输出到特定的out_dir文件夹中
index = 1
for (path, namedir, filenames) in os.walk(input_dir):
    # 对输入文件夹input_dir中的每一个jpg图像进行提取并输出到out_dir中。
    for filename in filenames:
        if filename.endswith('.jpg'):
            # 提取图像的路径
            img_path = path + '/' + filename
            img = cv2.imread(img_path)
            # 提取人脸的检测框，获得dets，dlib是对灰度图像进行处理的，故转为灰度图像
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_image, 1)
            # 使用enumerate 函数遍历序列中的元素以及它们的下标
            # 下标i即为人脸序号
            # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
            # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
            for i, d in enumerate(dets):
                y1 = d.top() if d.top() > 0 else 0
                y2 = d.bottom() if d.bottom() > 0 else 0
                x1 = d.left() if d.left() > 0 else 0
                x2 = d.right() if d.right() > 0 else 0
                # 在原彩色图像中截取face区域，注意不能是对灰度图像进行处理
                face = img[y1:y2, x1:x2]
                # img[y:y+h,x:x+w] 截取框内的人脸

                # 对截取的人脸图像进行resize，并显示输出
                face = cv2.resize(face, (size, size))
                cv2.imshow('image', face)

                # 将resize后的人脸写入out_dir文件夹中
                cv2.imwrite(out_dir + '/' + str(index) + '.jpg', face)
                index += 1
            # 设定终止key，按esc键停止
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
            # 按q键停止
            # if cv2.waitKey(30) & 0xff == ord('q'):
            #     sys.exit