import cv2
import os
import numpy as np 
import math
from math import *
import random

def seamlessClone(src_path, src2_path):
    src1 = cv2.imread(src_path)
    src2 = cv2.imread(src2_path)
    height, width, channels = src1.shape
    dim = (width, height)
    src2 = cv2.resize(src2, dim, interpolation = cv2.INTER_AREA)
    # Create an all white mask
    mask = 255 * np.ones(src1.shape, src1.dtype)

    # The location of the center of the src in the dst
    center = (width // 2, height // 2)

    # Seamlessly clone src into dst and put the results in output
    normal_clone = cv2.seamlessClone(src1, src2, mask, center, cv2.NORMAL_CLONE)
    #mixed_clone = cv2.seamlessClone(src1, src2, mask, center, cv2.MIXED_CLONE)
    return normal_clone

def calculate_points_from_mask(mask):
    margin = 50
    #copy make border for contours close to the edges
    expanded = cv2.copyMakeBorder(mask, margin, margin, margin, margin, cv2.BORDER_CONSTANT, None, value = 0) 
    thresh = cv2.threshold(expanded, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    points = get_corner_points_of_contour(contours[0])

    #remove the margin
    for point in points:
        point[0] = point[0] - margin
        point[1] = point[1] - margin

    return points

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_corner_points_of_contour(contour):
    points = []
    rect = cv2.minAreaRect(contour)
    center = rect[0]  # Center of the bounding rectangle

    top_left_point = None
    top_left_distance = 0

    top_right_point = None
    top_right_distance = 0

    bottom_left_point = None
    bottom_left_distance = 0

    bottom_right_point = None
    bottom_right_distance = 0

    for i in range(0, len(contour), 1):
        point = tuple(contour[i][0])  # Convert point to tuple (x, y)
        dist = distance(point, center)

        if point[0] < center[0] and point[1] < center[1]:
            if dist > top_left_distance:
                top_left_point = point
                top_left_distance = dist
        elif point[0] > center[0] and point[1] < center[1]:
            if dist > top_right_distance:
                top_right_point = point
                top_right_distance = dist
        elif point[0] < center[0] and point[1] > center[1]:
            if dist > bottom_left_distance:
                bottom_left_point = point
                bottom_left_distance = dist
        elif point[0] > center[0] and point[1] > center[1]:
            if dist > bottom_right_distance:
                bottom_right_point = point
                bottom_right_distance = dist

    points.append(top_left_point)
    points.append(top_right_point)
    points.append(bottom_right_point)
    points.append(bottom_left_point)

    return points

def merge_with_mask(src_path, src_mask_path, src2_path):
    src1 = cv2.imread(src_path)
    src2 = cv2.imread(src2_path)

    mask = cv2.imread(src_mask_path,cv2.IMREAD_GRAYSCALE)
    mask_inverted = cv2.bitwise_not(mask)

    src1_masked = cv2.bitwise_and(src1, src1, mask=mask)
    src2_masked = cv2.bitwise_and(src2, src2, mask=mask_inverted)

    dst = cv2.addWeighted(src1_masked, 1.0, src2_masked, 1.0, 0.0)

    points = calculate_points_from_mask(mask)
   
    return dst, points

def merge_with_synthesized(synth_doc_path,background_path):
    cloned = seamlessClone(synth_doc_path,"./data/background/white-desktop.jpg")

    background = cv2.imread(background_path)

    cloned_height, cloned_width = cloned.shape[:2]
    cloned_ratio = cloned_width/cloned_height

    height, width, channels = background.shape

    margin_x =  math.floor(width*0.1)

    scale_ratio = (width - margin_x * 2)/width
    resized_width = math.floor(width - margin_x * 2)
    resized_height = math.floor(resized_width/cloned_ratio)

    margin_y = math.floor((height - resized_height)/2)

    dim = (resized_width, resized_height)
    angle = random.uniform(-1,1)*10
    cloned = rotate_img(cloned, angle)
    cloned = cv2.resize(cloned, dim, interpolation = cv2.INTER_AREA)
    
    cloned = cv2.copyMakeBorder(cloned, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_CONSTANT, None, value = 0) 
    #keep the same size
    cloned = cv2.resize(cloned, (width,height), interpolation = cv2.INTER_AREA)

    mask = cv2.cvtColor(cloned, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(mask, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask = np.ones(cloned.shape, cloned.dtype)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    points = get_points_from_minarearect(rect)

    mask = cv2.drawContours(mask,contours,-1,(255,255,255),-1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_inverted = cv2.bitwise_not(mask)

    background_masked = cv2.bitwise_and(background, background, mask=mask_inverted)
    dst = cv2.addWeighted(cloned, 1.0, background_masked, 1.0, 0.0)
    return dst, points

def get_points_from_minarearect(rect):
    (center_x, center_y), (width, height), angle = rect

    angle_rad = np.radians(angle)

    dx = width / 2
    dy = height / 2

    corner_offsets = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy) 
    ]

    corners = []
    for offset in corner_offsets:
        x_rot = offset[0] * np.cos(angle_rad) - offset[1] * np.sin(angle_rad)
        y_rot = offset[0] * np.sin(angle_rad) + offset[1] * np.cos(angle_rad)
        corners.append((center_x + x_rot, center_y + y_rot))

    corners = np.int0(corners)
    return corners  

def rotate_img(img,angle = 3):
    rows, cols = img.shape[:2]
    center = (cols / 2, rows / 2)
    height_new = int(cols*fabs(sin(radians(angle)))+rows*fabs(cos(radians(angle))))
    width_new = int(rows*fabs(sin(radians(angle)))+cols*fabs(cos(radians(angle))))
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0,2] += (width_new-cols)/2  
    M[1,2] += (height_new-rows)/2
    dst = cv2.warpAffine(img, M, (width_new, height_new))
    return dst

def write_annotation(box, output_path):
    annotation = ""
    for point in box:
        annotation += str(point[0]) + "," + str(point[1]) + " "
    annotation = annotation.strip()
    with open(output_path, "w") as f:
        f.write(annotation)

if __name__ == "__main__":
    outdir = "./output"
    if os.path.exists("./output") == False:
        os.mkdir(outdir)
    #use photo synthesized
    for filename in os.listdir("./data/synthesized-doc"):
        src_path = "./data/synthesized-doc/" + filename
        for background_filename in os.listdir("./data/background"):
            src2_path = "./data/background/" + background_filename
            dst,box = merge_with_synthesized(src_path, src2_path)
            output_name = background_filename.split(".")[0] + "-" + filename
            print(output_name)
            write_annotation(box,outdir + "/" + output_name+".txt")
            cv2.imwrite(outdir + "/" + output_name, dst)

    #use photo taken
    for filename in os.listdir("./data/doc"):
        if filename.find("mask") == -1:
            src_path = "./data/doc/" + filename
            src_mask_path = "./data/doc/" + filename.split(".")[0] + "-mask.png"
            for background_filename in os.listdir("./data/background"):
                src2_path = "./data/background/" + background_filename
                dst,box = merge_with_mask(src_path, src_mask_path, src2_path)
                output_name = background_filename.split(".")[0] + "-" + filename
                print(output_name)
                write_annotation(box,outdir + "/" + output_name+".txt")
                cv2.imwrite(outdir + "/" + output_name, dst)
    
