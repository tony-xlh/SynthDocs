import cv2
import os
import numpy as np 
 
def merge(src_path, src_mask_path, src2_path):
    src1 = cv2.imread(src_path)
    src2 = cv2.imread(src2_path)

    mask = cv2.imread(src_mask_path,cv2.IMREAD_GRAYSCALE)
    mask_inverted = cv2.bitwise_not(mask)

    src1_masked = cv2.bitwise_and(src1, src1, mask=mask)
    src2_masked = cv2.bitwise_and(src2, src2, mask=mask_inverted)

    dst = cv2.addWeighted(src1_masked, 1.0, src2_masked, 1.0, 0.0)
    return dst

if __name__ == "__main__":
    outdir = "./output"
    if os.path.exists("./output") == False:
        os.mkdir(outdir)
    for filename in os.listdir("./data/doc"):
        if filename.find("mask") == -1:
            src_path = "./data/doc/" + filename
            src_mask_path = "./data/doc/" + filename.split(".")[0] + "-mask.png"
            for background_filename in os.listdir("./data/background"):
                src2_path = "./data/background/" + background_filename
                dst = merge(src_path, src_mask_path, src2_path)
                output_name = background_filename.split(".")[0] + "-" + filename
                print(output_name)
                cv2.imwrite(outdir + "/" + output_name, dst)
