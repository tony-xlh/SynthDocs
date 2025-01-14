import cv2
import os
import numpy as np 
import math
 
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
    print(center)
    # Seamlessly clone src into dst and put the results in output
    normal_clone = cv2.seamlessClone(src1, src2, mask, center, cv2.NORMAL_CLONE)
    #mixed_clone = cv2.seamlessClone(src1, src2, mask, center, cv2.MIXED_CLONE)
    return normal_clone

def merge_with_mask(src_path, src_mask_path, src2_path):
    src1 = cv2.imread(src_path)
    src2 = cv2.imread(src2_path)

    mask = cv2.imread(src_mask_path,cv2.IMREAD_GRAYSCALE)
    mask_inverted = cv2.bitwise_not(mask)

    src1_masked = cv2.bitwise_and(src1, src1, mask=mask)
    src2_masked = cv2.bitwise_and(src2, src2, mask=mask_inverted)

    dst = cv2.addWeighted(src1_masked, 1.0, src2_masked, 1.0, 0.0)
    return dst

def merge_with_synthesized(synth_doc_path,background_path):
    cloned = seamlessClone(synth_doc_path,"./data/background/white-desktop.jpg")
    cv2.imwrite("cloned.jpg", cloned)
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

    cloned = cv2.resize(cloned, dim, interpolation = cv2.INTER_AREA)

    mask = 255 * np.ones(cloned.shape, cloned.dtype)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    cloned = cv2.copyMakeBorder(cloned, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_CONSTANT, None, value = 0) 
    mask = cv2.copyMakeBorder(mask, margin_y, margin_y, margin_x, margin_x, cv2.BORDER_CONSTANT, None, value = 0) 
    
    #keep the same size
    cloned = cv2.resize(cloned, (width,height), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (width,height), interpolation = cv2.INTER_AREA)
    
    mask_inverted = cv2.bitwise_not(mask)
    
    background_masked = cv2.bitwise_and(background, background, mask=mask_inverted)
    dst = cv2.addWeighted(cloned, 1.0, background_masked, 1.0, 0.0)
    return dst
    

if __name__ == "__main__":
    outdir = "./output"
    if os.path.exists("./output") == False:
        os.mkdir(outdir)
    #use photo synthesized
    for filename in os.listdir("./data/synthesized-doc"):
        src_path = "./data/synthesized-doc/" + filename
        for background_filename in os.listdir("./data/background"):
            src2_path = "./data/background/" + background_filename
            dst = merge_with_synthesized(src_path, src2_path)
            output_name = background_filename.split(".")[0] + "-" + filename
            print(output_name)
            cv2.imwrite(outdir + "/" + output_name, dst)
    #use photo taken
    for filename in os.listdir("./data/doc"):
        if filename.find("mask") == -1:
            src_path = "./data/doc/" + filename
            src_mask_path = "./data/doc/" + filename.split(".")[0] + "-mask.png"
            for background_filename in os.listdir("./data/background"):
                src2_path = "./data/background/" + background_filename
                dst = merge_with_mask(src_path, src_mask_path, src2_path)
                output_name = background_filename.split(".")[0] + "-" + filename
                print(output_name)
                cv2.imwrite(outdir + "/" + output_name, dst)
    
