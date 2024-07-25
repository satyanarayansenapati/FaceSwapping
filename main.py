from PIL import Image
import cv2
from swap import logger
import os
from swap.functions import image_face, face_data, face_swapping



# getting the source image
src_image_path = str(input("Please write the path of source image on which swapping will be performed\n"))
logger.info(f"SOURCE IMAGE : {src_image_path} has been provided to fetch the image")

# checking the input
if os.path.exists(src_image_path) and src_image_path[-3]!= 'jpg' or 'png':
    
    # reading the image data from the path
    src_image,src_face_data = image_face(src_image_path)

    # displaying the main image
    main_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    main_image = Image.fromarray(main_image)
    main_image.show()

else :
    logger.error(f"User selected an invalid file or path")
    print("Please enter a valid path and file")
    

# extracting the face data from the source image
source_image_face_data = face_data(image=src_image,faces=src_face_data)


# displaying the options to select face to be swapped
for i in range(len(source_image_face_data.keys())):
    im = source_image_face_data[f"person_{i}"]['crop_face_image']
    im.show()
    print(f"person_{i}")


selected_faces = str(input("Please enter the person ids that you want to be swapped with comma separated")).split(',')

# target face image input
# it is recommended to choose a taget photo where the number of face = 1
tar_image_path = str(input("Please write the path of target image of which face will be used to for swapping\n"))

logger.info(f"TARGET IMAGE : {tar_image_path} has been provided to fetch the image")

# checking the input
if os.path.exists(tar_image_path) and tar_image_path[-3]!= 'jpg' or 'png':
    
    # reading the image data from the path
    tar_image,tar_face_data = image_face(tar_image_path)
    target_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
    target_image = Image.fromarray(target_image)
    target_image.show()
    
else :
    logger.error(f"User selected an invalid file or path")
    print("Please enter a valid path and file")

# extracting the face data from the source image
target_image_face_data = face_data(image=tar_image,faces=tar_face_data)




# swapping
output_image = face_swapping(selected_face=selected_faces,source_image=src_image,face_data=source_image_face_data,swap_face=target_image_face_data)
output_image = Image.fromarray(output_image)


# saving the output image
output_image.save('output.jpg')

output_image.show()
print("This image can be found in output folder")