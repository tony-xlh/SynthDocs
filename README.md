# SynthDocs

Code for generating synthetic document images for edge detection.

It can blend documents with different backgrounds to create new document images.

It uses two kinds of document images. One is taken in a real scene. The other is pure document images from datasets like [DocBank](https://github.com/doc-analysis/DocBank).

You can find the generated dataset here: https://github.com/tony-xlh/SynthDocs/releases/tag/dataset

## How the Data is Collected

The document content and the background can affect the final document detection result. So we've taken photos of different types of documents (receipts, ID cards and A4 documents) on a background with various patterns and colors.

## How the Blending of Images Work

1. Blend a document image taken in a real scene and a background image.

    We have a background image, a document image, and a mask image of the document.

    ![src2](https://github.com/user-attachments/assets/d2368166-bfd5-4789-aaca-960087c25859)
   
    ![src1](https://github.com/user-attachments/assets/408b72b4-fc0f-4d4c-bcdb-86018abcfb5c)
   
    ![mask](https://github.com/user-attachments/assets/4aae94ec-e64d-45bc-9f7f-488bd14c1f19)

    Here are the steps:
   
    1. Create an image containing the document using OpenCV's `bitwise_and` method and the mask image.
  
       ![src1_mask](https://github.com/user-attachments/assets/3afb6054-3371-472d-8dd5-23e4f02ce059)

    2. Create an image containing the background using OpenCV's `bitwise_and` method and the inverted mask image.
  
       ![src2_mask](https://github.com/user-attachments/assets/b33d52e7-a117-43f4-8bdc-79c70f2485ff)

    3. Blend the two images together to create the final result using OpenCV's `addWeighted` method.
  
       ![dst](https://github.com/user-attachments/assets/71bcaf0b-5d10-47f8-b25e-5285e6325bef)


    
3. Blend a document image that covers 100% of the image and a background image.

   We have the following image to blend:

   ![src1](https://github.com/user-attachments/assets/704b3c0d-9feb-42ff-aecc-13f2eb941652)

   Here are the steps:

   1. Use OpenCV's `seamlessClone` method and a plain background image to make the doc like a photo taken in a real scene.
  
       Result:
      
       ![normal_clone](https://github.com/user-attachments/assets/aba7bbef-eb7e-4f5a-9269-308c838cd617)

       Background image:
      
       ![src2](https://github.com/user-attachments/assets/5daf72d5-89d1-41ff-abd1-8d4eeeb476a0)

   3. Rotate the image with a random degree and add padding to make it match the background image.
  
      ![cloned](https://github.com/user-attachments/assets/356e94c5-6114-4ba6-972e-4b723949f39c)

   4. Do similar steps for the previous document image type to blend the two images together.
  
      Result:

      ![black-desktop-10 tar_1701 04170 gz_TPNL_afterglow_evo_8](https://github.com/user-attachments/assets/c28bc5f7-ce6e-44e2-b631-c083047f530c)

## Annotation

The annotation is calculated based on the mask image and stored in the following format:

```
x1,y1 x2,y2 x3,y3 x4,y4
```

## Related

* [SmartDoc 2015 – Challenge 1 SMARTPHONE DOCUMENT CAPTURE COMPETITION](http://smartdoc.univ-lr.fr/smartdoc-2015-challenge-1/)
* [VIDEO DOCUMENT SCANNING](https://arxiv.org/pdf/2411.00576)
* [Click-Free, Video-Based Document Capture](http://faculty.pucit.edu.pk/nazarkhan/work/cfdc/cfdc_cbdar17.pdf)
* [HU-PageScan: a fully convolutional neural network for document page crop](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ipr.2020.0532)








