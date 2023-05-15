import os 
import glob
from PIL import Image
import cv2


""""
# the path to the original images 
DIR  = "../../rawdata/images_2d/D14"
DEST = "../../rawdata/test2/"

for ele in os.listdir(DIR):
    # covert the images from .TIF to .png
    #tif_image = Image.open(os.path.join(DIR,ele))
    #img = cv2.imread('path/to/image.tif', cv2.IMREAD_ANYDEPTH)
    tif_image = cv2.imread(os.path.join(DIR,ele),cv2.IMREAD_ANYDEPTH)
    new_name  = ele.replace(".tif",".png")
    image = cv2.cvtColor(tif_image, cv2.COLOR_BGR2RGB)
    #tif_image = tif_image.convert('RGB')
    print(tif_image)
    cv2.imwrite(os.path.join(DEST,new_name), tif_image)
    #tif_image.save(os.path.join(DEST,new_name),"PNG")
"""


input_dir  = "../../rawdata/images_2d_png/D14"
output_dir = "../../rawdata/test15"
rows = 6
cols = 6
image_types = ["CBP","IMR","EP300"]

for img_type in image_types:

    # create an output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # create an output dir + img_type
    if not os.path.exists(os.path.join(output_dir,img_type)):
        os.makedirs(os.path.join(output_dir,img_type))

    for filename in os.listdir(os.path.join(input_dir,img_type)):


        
        img = Image.open(os.path.join(input_dir,img_type,filename))
        # Define a threshold for the proportion of dark pixels
        dark_threshold = 2
        # Get the size of the image
        width, height = img.size
        # Calculate the width and height of each sub-image
        sub_width = width // cols
        sub_height = height // rows
        # Loop through each row and column in the grid
        for row in range(rows):
            for col in range(cols):
                # Calculate the coordinates of the sub-image
                left = col * sub_width
                top = row * sub_height
                right = (col + 1) * sub_width
                bottom = (row + 1) * sub_height

                # Crop the sub-image
                sub_img = img.crop((left, top, right, bottom))

                # Get the histogram of the sub-image
                histogram = sub_img.histogram()

                # Calculate the proportion of dark pixels
                dark_pixels = sum(histogram[:64])+sum(histogram[256:320])+sum(histogram[512:576])
                total_pixels = sum(histogram)
                dark_proportion = dark_pixels / total_pixels

                # Check if the sub-image has a high proportion of dark pixels
                if dark_proportion < dark_threshold:
                    output_filename = f'{filename.replace(".png","")}_{row}_{col}.png'
                    sub_img.save(os.path.join(output_dir,img_type,output_filename))