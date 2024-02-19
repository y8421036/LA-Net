"""
1、6M把GT逆时针旋转270度；3M逆时针旋转90
2、把背景像素设为0，FAZ的设为1，RV的设为2
"""

# -*- coding: UTF-8 -*-
import sys

import numpy
from PIL import Image
import os


def get_filelist(path): 
    fileList = []
    for home, dirs, files in os.walk(path):
        for fileName in files:
            fileList.append(os.path.join(path, fileName))

    return fileList


if __name__ == "__main__":
    n_classes = 2  
    if n_classes == 3:
        rv_pixel_value = 2  
        faz_pixel_value = 1  
    elif n_classes == 2:
        rv_pixel_value = 1  
        faz_pixel_value = 0  
    else:
        print('class number error')
        sys.exit(0)

    filePath = '/data2/datasets/OCTA-500/3M/GroundTruth'
    outputPath = '/data2/datasets/OCTA-500/3M/Label_RV'
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    Filelist = get_filelist(filePath)

    for filename in Filelist:
        print(filename)
        im = Image.open(filename)

        im = numpy.array(im)
        im_replace = numpy.where(im == 255, rv_pixel_value, im)
        im_replace = numpy.where(im_replace == 100, faz_pixel_value, im_replace)

        im_replace = Image.fromarray(im_replace)

        im_rotate = im_replace.transpose(Image.ROTATE_90) 

        output_path = filename.replace(filePath, outputPath)
        im_rotate.save(output_path)
