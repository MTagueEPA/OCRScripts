import io
import imutils

import cv2
import deskew
import fitz
import numpy as np
from PIL import Image
from tesserocr import OEM, PSM, RIL, PyTessBaseAPI

def get_pages(file, zoom=3):
    """
    Open PDF and return a list of pages as numpy arrays.
    Args:
        file: str, PDF file
        zoom: int, optional, zoom factor of 3 in each dimension, defaults to 3
    Returns:
        image_pgs: list, list of images
    """

    # zoom factor of 3 in each dimension
    mat = fitz.Matrix(zoom, zoom)
    # image list
    image_pgs = []
    # open document
    doc = fitz.open(file)

    # iterate through the pages
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        data = pix.tobytes()
        img = Image.open(io.BytesIO(data))
        img.convert("RGB")
        image_pgs.append(img)

    return image_pgs


def ocr_image_to_word_list(image):
    """
    This function takes an image as input, performs OCR on it using the PyTessBaseAPI,
    and returns a list of dictionaries, where each dictionary represents a word in the image.
    Args:
        image: A PIL Image object
    Returns:
        word_list: A list of dictionaries, where each dictionary represents a word in the image
    """
    word_list = []
    span_num = 0
    # Initialize the PyTessBaseAPI with the specified language, PSM and OEM
    with PyTessBaseAPI(lang='eng', psm=PSM.SINGLE_COLUMN, oem=OEM.DEFAULT) as api:
        level = RIL.WORD
        # Provide the image to the API
        api.SetImage(image)
        api.Recognize()
        ri = api.GetIterator()
        while True:
            try:
                word = ri.GetUTF8Text(level)
                box = ri.BoundingBox(level)
                word_dict = {"text": word, "bbox": [*box], "flags": 0, "block_num": 0, "line_num": 0, "span_num": span_num}
                word_list.append(word_dict)
                span_num += 1
                if not ri.Next(level):
                    break
            except:
                return word_list
    return word_list

    
def orientation_and_deskew(image):
    """
    Detect image orientation and rotate if needed.
    Args:
        image: array, image of pdf page
    Returns:
        image: PIL image, rotated image
    """
    orientation_dict = {0:0, 1:90, 2:180, 3:270}
    with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
        api.SetImage(image)
        api.Recognize()
        it = api.AnalyseLayout()
        orientation, _, _, _ = it.Orientation()
    image = PIL_to_cv(image)
    image = imutils.rotate_bound(image, angle=360-orientation_dict[orientation])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    angle = deskew.determine_skew(image)
    image = imutils.rotate_bound(image, angle=-angle)
    return cv_to_PIL(image)


def process_text_page(image):
    """
    Process a text page for better ocr results.
    Parameters:
        image: array, image of a text page
    Returns:
        proc_img: PIL image, processed text page image
    """
    # image = sharpen_image(image)
    # image = binarizeBlur_image(image)
    image = PIL_to_cv(image)
    # convert image from RGB to GRAY
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 1x1 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # dilate, erode, and threshold the image
    proc_img = cv2.dilate(image, kernel, iterations=1)
    proc_img = cv2.erode(~proc_img, kernel, iterations=1)
    thresh, proc_img = cv2.threshold(
        proc_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return cv_to_PIL(proc_img)


def PIL_to_cv(pil_img):
    """
    Convert PIL image to openCV image format
    :param pil_img (PIL.Image): PIL image to be converted
    :return: numpy.ndarray openCV image format
    """
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv_to_PIL(cv_img):
    """
    Convert a cv2 image to PIL image
    :param cv_img: input cv2 image
    :return: PIL image
    """
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
