import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

def preprocess(img):

    img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    img_blur=cv.GaussianBlur(img_gray,(7,7),0)

    img_thresh=cv.adaptiveThreshold(img_blur, 255, 1,1, 11, 2)

    # img_inverted= cv.bitwise_not(img_thresh, 0)
    #
    # kernel= cv.getStructuringElement(cv.MORPH_RECT, (1,1))
    #
    # img_morph = cv.morphologyEx(img_inverted, cv.MORPH_OPEN, kernel)
    #
    # img_dilate=cv.dilate(img_morph,kernel,iterations=1)

    preprocessed_img=img_thresh

    return preprocessed_img

def biggest_contour(contours):
    main_points=np.array([])
    max_area=0
    for i in contours:
        area=cv.contourArea(i)
        if area>50:
            peri=cv.arcLength(i,True)
            points=cv.approxPolyDP(i,0.02*peri,True)
            if area>max_area and len(points)==4:
                max_area=area
                main_points=points
    return main_points,max_area

def reorder(points):
    points=points.reshape((4,2))
    points_new=np.zeros((4,1,2),dtype=np.int32)
    add=points.sum(1)
    points_new[0]=points[np.argmin(add)]
    points_new[3]=points[np.argmax(add)]
    diff=np.diff(points,axis=1)
    points_new[1]=points[np.argmin(diff)]
    points_new[2]=points[np.argmax(diff)]
    return points_new

def split_into_squares(img):
    rows=np.vsplit(img,9)
    boxes=[]
    for row in rows:
        cols=np.hsplit(row,9)
        for box in cols:
            boxes.append(box)
    return boxes

def intializePredictionModel():
    model=load_model("final_model.hdf5")
    return model

def getPrediction(boxes,model):
    result=[]
    for image in boxes:

        img=cv.resize(image,(32,32))
        #img=preprocess(img)#marked
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img = cv.GaussianBlur(img, (3, 3), 0)
        # img_thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        img = tf.keras.utils.normalize(img, axis=1)
        img=np.array(img).reshape(-1,32,32,1)
        #prediction
        prediction=model.predict(img)
        class_Index=np.argmax(prediction,axis=-1)
        prob_value=np.amax(prediction)
        print(class_Index+1,prob_value)
        #save the result
        if prob_value > 0.85:
            result.append(class_Index[0]+1)
        else:
            result.append(0)
    return result
########################################################################
# def process(img):#marked
#
#     img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#
#     img_blur=cv.GaussianBlur(img_gray,(3,3),0)
#
#     img_thresh=cv.adaptiveThreshold(img_blur, 255, 1,1, 11, 2)
#
#     #img_inverted= cv.bitwise_not(img_thresh, 0)
#
#     kernel= cv.getStructuringElement(cv.MORPH_RECT, (1,1))
#
#     img_morph = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)
#
#     img_dilate=cv.dilate(img_morph,kernel,iterations=1)
#
#     processed_img=img_dilate
#
#     return processed_img
################################################################################

def get_grid_lines(img, length=10):
    horizontal = grid_line_helper(img, 1, length)
    vertical = grid_line_helper(img, 0, length)
    return vertical, horizontal

def grid_line_helper(img, shape_location, length=10):
    clone = img.copy()
    # if its horizontal lines then it is shape_location 1, for vertical it is 0
    row_or_col = clone.shape[shape_location]
    # find out the distance the lines are placed
    size = row_or_col // length

    # find out an appropriate kernel
    if shape_location == 0:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, size))
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, 1))

    # erode and dilate the lines
    clone = cv.erode(clone, kernel)
    clone = cv.dilate(clone, kernel)

    return clone

def create_grid_mask(vertical, horizontal):
    # combine the vertical and horizontal lines to make a grid
    grid = cv.add(horizontal, vertical)
    # threshold and dilate the grid to cover more area
    grid = cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 235, 2)
    grid = cv.dilate(grid, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=2)

    # find the list of where the lines are, this is an array of (rho, theta in radians)
    pts = cv.HoughLines(grid, .3, np.pi / 90, 200)

    lines = draw_lines(grid, pts)
    # extract the lines so only the numbers remain
    mask = cv.bitwise_not(lines)
    return mask

def draw_lines(img, lines):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    clone = img.copy()
    lines = np.squeeze(lines)

    for rho, theta in lines:
        # find out where the line stretches to and draw them
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 4)
    return clone

def clean_helper(img):
    # print(np.isclose(img, 0).sum())
    if np.isclose(img, 0).sum() / (img.shape[0] * img.shape[1]) >= 0.95:
        return np.zeros_like(img), False

    # if there is very little white in the region around the center, this means we got an edge accidently
    height, width = img.shape
    # mid = width // 2
    # if np.isclose(img[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (2 * width * 0.4 * height) >= 0.90:
    #     return np.zeros_like(img), False

    # center image
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    x, y, w, h = cv.boundingRect(contours[0])

    start_x = (width - w) // 2
    start_y = (height - h) // 2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]

    return new_img, True

def clean_squares(squares):
    cleaned_squares = []
    i = 0

    for square in squares:
        new_img, is_number = clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
            i += 1

        else:
            cleaned_squares.append(0)

    return cleaned_squares

def display_numbers(img,numbers,color=(0,0,255)):
    secW=int(img.shape[0]/9)
    secH=int(img.shape[1]/9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x]!=0:
                cv.putText(img,str(numbers[(y*9)+x]),(x*secW+int(secW/2)-10,int((y+0.8)*secH)),cv.FONT_HERSHEY_PLAIN,
                           2,color,2,cv.LINE_AA)
    return img
