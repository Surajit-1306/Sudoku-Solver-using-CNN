import numpy as np
import cv2 as cv
from testing import *
import tensorflow as tf
from soduku_solver import SudokuSolver
import time
import tkinter as tk
from tkinter import filedialog




##########################################
# Create the main application window
root = tk.Tk()
root.title("Sudoku Solver")

# Image display labels
original_image_label = tk.Label(root)
solved_image_label = tk.Label(root)
##########################################
# Function to upload and process the image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv.imread(file_path)
    #########################################
    model=intializePredictionModel()
    width=450
    height=450

    #########################################
    start_time=time.time()
    #img= cv.imread(path)
    img=cv.resize(img,(width,height),interpolation=cv.INTER_AREA)

    img_blank=np.zeros((height,width,3),np.uint8)

    img_thresh=preprocess(img)

    img_contours=img.copy()
    contours, hierchy = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_contours,contours,-1,(0,355,255),3)
    main_points, max_area=biggest_contour(contours)

    if main_points.size!=0:
        main_points=reorder(main_points)
        img_bigcontour=cv.drawContours(img_contours,main_points,-1,(0,0,255),12)
        pts1=np.float32(main_points)
        pts2=np.float32([(0,0),(width,0),(0,height),(width,height)])
        matrix=cv.getPerspectiveTransform(pts1,pts2)
        img_warped=cv.warpPerspective(img,matrix,(width,height))


        img_warped=preprocess(img_warped)

        vertical_lines, horizontal_lines = get_grid_lines(img_warped)
        mask = create_grid_mask(vertical_lines, horizontal_lines)
        numbers = cv.bitwise_and(img_warped, mask)
        #
        boxes=split_into_squares(numbers)
        boxes=clean_squares(boxes)

        numbers=getPrediction(boxes,model)

        img_display_num=img_blank.copy()
        img_display_num=display_numbers(img_display_num,numbers,color=(0,255,0))
        numbers=np.asarray(numbers)
        print(numbers)
        position_val=np.where(numbers>0,0,1)
        print(position_val)

        board=np.array_split(numbers,9)
        #print(board)
        solver = SudokuSolver(board)
        try:
            board=solver.solve()
        except:
            pass

        board=np.asarray(board)

        flat_list=board * position_val
        print(flat_list)
    else:
        print("Please give a clear Sudoku image.")
    img_solved_num=img_blank.copy()
    img_solved_num=display_numbers(img_solved_num,flat_list,color=(128,277,0))

    pts2 = np.float32(main_points)
    pts1 = np.float32([(0, 0), (width, 0), (0, height), (width, height)])

    homography_matrix, _ = cv.findHomography(pts1, pts2)
    result_img = cv.warpPerspective(img_solved_num, homography_matrix, (img.shape[1], img.shape[0]))
    final_image = cv.addWeighted(img, 0.5, result_img, 1.25, 0)

    cv.imshow('img1',final_image)
    end_time=time.time()
    elapsed_time=end_time-start_time
    print(f"Total time: {round(elapsed_time,2)} seconds")

    cv.waitKey(0)

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()
root.mainloop()
