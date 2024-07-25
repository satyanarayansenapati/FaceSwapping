# imports
import numpy as np
import os
import glob
import cv2

import insightface
from insightface.app import FaceAnalysis
from PIL import Image

from swap import logger

# entites

# face detection app
app = FaceAnalysis(name = "buffalo_l")
app.prepare(ctx_id = 0, det_size = (640,640))

# swapper app
swapper = insightface.model_zoo.get_model(
    "model_file/inswapper_128_faceswap.onnx",
    download = False,
    download_zip = False
)



# functions 
def image_face(filepath:str):

    '''
    args : The filepath where the image on which swapping will be performed is stored
    output : The image and the faces in the image
    
    '''

    logger.info(f"The provided path for image is : {filepath}")
    
    # reading the image
    try:

        image = cv2.imread(filepath)

    except Exception as e:

        logger.error(f"Function : get_face\nStatus: Failed to read the image at {filepath}\nError : {e}")

    # storing the faces in the image in a variable
    try:
        faces = app.get(image)
    
    except Exception as e:
        logger.error(f"Function : get_face\nStatus: Failed to extract faces in the image\nError : {e}")

    return image, faces



# face data extraction from the found faces
# function to store source_image, person's face, cropped_face
def face_data_extractor(image,person):

    '''
    It is sub function
    
    args : source image, the data of all the faces present in the image
    output : a dictorany containing the source image, person's face data, cropped face image(for pillow)
    
    '''

    # getting the face
    person_face = person

    # saving cropped face image
    try:
        face_area = person['bbox']
        im = Image.fromarray(image)
        crop_face = im.crop(face_area)

        logger.info(f"Function : face_data_extractor\Status : Success")

    except Exception as e:
        logger.error(f"Function : face_data_extractor\nStatus : Error during creation of crop face image\nError : {e}")
    

    data = {'source_image': image, 'person_face': person_face, 'crop_face_image': crop_face}

    return data



# function to return all the data read from the image
def face_data(image,faces):

    '''
    args: source image, image data about faces
    return : a dictonary containing source image, person's face data, cropped face image(for pillow) for each face found in the image'''

    # color correction
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info("Color correction performed")

    # checking if no face was found in the image
    if len(faces) == 0:
        logger.info("No face found")
        
    
    # saving the number of faces found in a variable
    no_of_faces = len(faces)
    logger.info(f"No of faces found in the image : {no_of_faces}")
    
    try:
        person_data = {}
        
        # extracting face information from the dictonary
        for i in range(no_of_faces):
            keys = f"person_{i}"
            person_data[keys] = face_data_extractor(image = image, person = faces[i])

        logger.info("Face data is generated and saved successfully")

    except Exception as e:
        logger.error(f"Function : face_data\nStatus : Error during storing the face data\nError : {e}")
    
    return person_data


# main swapping function
def face_swapping(selected_face:list,source_image,face_data, swap_face):
    '''
    args:
        selected_face : (list) the keys of the faces that would be swapped
        source_image : (list) the image on which swapping will be done
        face_data : (dictonary) data of all the faces from the source_image
        swap_face : the face which will replace the selected_face(s)
        
    output: an image 
    
    '''
    # checking the input of user
    if not set(selected_face) <= set(list(face_data.keys())):
        print("Please enter valid person id")

    swap_face = swap_face['person_0']['person_face']
    # creating a copy of the source image
    output_image = source_image.copy()


    if len(selected_face) == 0:
        print("Please select the faces you want to perform swapping")
    else:

        try:
            for i in selected_face:
                output_image = swapper.get(output_image,face_data[i]['person_face'],swap_face, paste_back = True)
            
            logger.info(f"Function : face_swapping\nStatus : Success")
        
        except Exception as e:
            logger.error(f"Function : face_swapping\nStatus : Error encountered while swapping faces\nError : {e}")
    
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    return output_image