#!/usr/bin/env python
# coding: utf-8
#filter_utils.py
# In[1]:


import os
import cv2

def apply_filters(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Histogram Equalization
    equalized = cv2.equalizeHist(blurred)

    # Apply Adaptive Thresholding
    #thresholded = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return equalized

def process_images(input_folder, output_folder):
    img_files = os.listdir(input_folder)

    for i, file_name in enumerate(img_files):
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)

        filtered_img = apply_filters(img)

        save_path = os.path.join(output_folder, file_name)

        cv2.imwrite(save_path, filtered_img)
    print("이미지 처리 완료")

def get_filtered_image_names(dir_path):
    file_list = [file_name for file_name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file_name))]
    
    data_names = []
    for file_name in file_list:
        data_name, extension = os.path.splitext(file_name)
        data_names.append(data_name)
    
    return data_names

