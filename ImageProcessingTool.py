import cv2
import numpy as np
from tkinter import *
import tkinter.messagebox
from matplotlib import pyplot as plt

# Creating Main Window
app = Tk()
app.title('Image Processing Tool')
app.geometry('980x500')
# ENTRY
label_entry = Label(app, text='Image and Video Entry', font="Times, 14", fg='red')
label_entry.place(x=30, y=5)
# Image Enrty
part_text1 = StringVar()
part_label1 = Label(app, text='Image Name', font="Times, 13")
part_label1.place(x=5, y=40)
part_entry = Entry(app, textvariable=part_text1)
part_entry.place(x=135, y=45)
# Video Enrty
part_text2 = StringVar()
part_label1 = Label(app, text='Video Name', font="Times, 13")
part_label1.place(x=5, y=80)
part_entry = Entry(app, textvariable=part_text2)
part_entry.place(x=135, y=85)
warning_label = Label(app, text='*Type the file name with the extension (.jpg/.png/.mp4)*', font="Times, 10")
warning_label.place(x=10, y=115)
# Functions for opening image and video / And their buttons
def open_image():
    if part_text1.get() != '':
        image1 = cv2.imread(part_text1.get())
        cv2.imshow('IMAGE-1', image1)
open_button1 = Button(app, text='Open Image', font="Times, 10", command=open_image)
open_button1.place(x=50, y=150)
def open_video():
    if part_text2.get() != '':
        cap = cv2.VideoCapture(part_text2.get())
        while True:
            ret, frame = cap.read()
            cv2.imshow('Original Video', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
open_button2 = Button(app, text='Open Video', font="Times, 10", command=open_video)
open_button2.place(x=155, y=150)
# Saving the image with asking some questions to user
def save_image(image):
    answer = tkinter.messagebox.askquestion('SAVE THE DOCUMENT', 'Do you want to save the image/video ?')
    if answer == 'yes':
        answer2 = tkinter.simpledialog.askstring("Name of the File",
                                                 "Which name do you want to give for file (example.jpg/example.png)?")
        if answer2 is not None:
            cv2.imwrite(answer2, image)
    else:
        tkinter.messagebox.showinfo('Return', 'You will now return to the application screen')
# Getting input for gamma transformation from user
part_text_gamma = StringVar()
part_label_gamma = Label(app, text='Gamma Value', font="Times, 11")
part_label_gamma.place(x=415, y=195)
height_entry = Entry(app, textvariable=part_text_gamma)
height_entry.place(x=520, y=197)
# Image Resize Parts
label_size = Label(app, text='Image Resize', font="Times, 14", fg='red')
label_size.place(x=405, y=5)
part_text3 = StringVar()
part_label2 = Label(app, text='Image Height', font="Times, 13")
part_label2.place(x=350, y=35)
height_entry = Entry(app, textvariable=part_text3)
height_entry.place(x=470, y=40)
part_text4 = StringVar()
part_label3 = Label(app, text='Image Width', font="Times, 13")
part_label3.place(x=350, y=75)
width_entry = Entry(app, textvariable=part_text4)
width_entry.place(x=470, y=80)
# Image Rotate Parts
label_rotate = Label(app, text='Image Rotate', font="Times, 14", fg='red')
label_rotate.place(x=405, y=115)
part_text5 = StringVar()
label_rotate2 = Label(app, text='Degree', font="Times, 13")
label_rotate2.place(x=370, y=145)
rotate_entry = Entry(app, textvariable=part_text5)
rotate_entry.place(x=470, y=150)
# Image Crop Parts
label_crop = Label(app, text='Image Crop', font="Times, 14", fg='red')
label_crop.place(x=710, y=5)
part_text6 = StringVar()
part_text7 = StringVar()
part_text8 = StringVar()
part_text9 = StringVar()
label_min_height = Label(app, text='Min Height', font="Times, 13")
label_min_height.place(x=650, y=35)
crop_entry1 = Entry(app, textvariable=part_text6)
crop_entry1.place(x=750, y=40)
label_max_height = Label(app, text='Max Height', font="Times, 13")
label_max_height.place(x=650, y=75)
crop_entry2 = Entry(app, textvariable=part_text7)
crop_entry2.place(x=750, y=80)
label_min_width = Label(app, text='Min Width', font="Times, 13")
label_min_width.place(x=650, y=115)
crop_entry3 = Entry(app, textvariable=part_text8)
crop_entry3.place(x=750, y=120)
label_max_width = Label(app, text='Max Width', font="Times, 13")
label_max_width.place(x=650, y=155)
crop_entry4 = Entry(app, textvariable=part_text9)
crop_entry4.place(x=750, y=160)
tkvar1 = StringVar()
tkvar2 = StringVar()
tkvar3 = StringVar()
tkvar4 = StringVar()
choices1 = {'Mean', 'Gaussian', 'Median', 'Laplacian', 'Bilateral', 'Averaging', 'Kernel Blurring',
            'Sobel CV_8U', 'Sobel CV_64F', 'Edge Preserving Filter'}
choices2 = {'Resizing', 'Rotation', 'Cropping', 'Perspective Transformation', 'Affine Transformation'}
choices3 = {'Image Negatives', 'Log Transformation', 'Gamma Transformation', 'Piecewise Transformation'}
choices4 = {'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat', 'Crossing',
            'Ellipse', 'Rectangular'}
tkvar1.set('Filtering')
tkvar2.set('Image Manipulation')
tkvar3.set('Intensity Transformation')
tkvar4.set('Morphological Transformation')
# Function for Piecewise Transformation
def pixelval(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        return (s1 / r1) * pix
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2
# Function for Edge Preserving Filter
def shift_filter(dest_image):
    img_shift_filtered = cv2.pyrMeanShiftFiltering(dest_image, 10, 50)
    return img_shift_filtered
# Dropdown choices for filtering
def do_filter():
    if tkvar1.get() == 'Mean':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get(), 1)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            figure_size = 9
            img_mean = cv2.blur(image, (figure_size, figure_size))
            img_mean = cv2.resize(img_mean, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Mean Filter", img_mean)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_mean)
    if tkvar1.get() == 'Gaussian':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get(), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            figure_size = 9
            img_gaussian = cv2.blur(image, (figure_size, figure_size))
            img_gaussian = cv2.resize(img_gaussian, (550, 500))
            image = cv2.resize(image, (550, 500))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            img_gaussian = cv2.cvtColor(img_gaussian, cv2.COLOR_HSV2RGB)
            cv2.imshow('Original Image', image)
            cv2.imshow("Gaussian Filter", img_gaussian)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_gaussian)
    if tkvar1.get() == 'Median':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get(), 1)
            figure_size = 9
            img_median = cv2.medianBlur(image, figure_size)
            img_median = cv2.resize(img_median, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Median Filter", img_median)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_median)
    if tkvar1.get() == 'Laplacian':
        ddepth = cv2.CV_16S
        kernel_size = 3
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            # Remove noise by blurring with a Gaussian filter
            image = cv2.GaussianBlur(image, (3, 3), 0)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_laplacian = cv2.Laplacian(image_gray, ddepth, ksize=kernel_size)
            img_laplacian_last = cv2.convertScaleAbs(img_laplacian)
            img_laplacian_last = cv2.resize(img_laplacian_last, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('Laplacian', img_laplacian_last)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_laplacian)
    if tkvar1.get() == 'Bilateral':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            img_bilateral = cv2.bilateralFilter(image, 9, 75, 75)
            img_bilateral = cv2.resize(img_bilateral, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('Bilateral', img_bilateral)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_bilateral)
    if tkvar1.get() == 'Averaging':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            img_averaged = cv2.blur(image, (5, 5))
            img_averaged = cv2.resize(img_averaged, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('Averaging', img_averaged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_averaged)
    if tkvar1.get() == 'Kernel Blurring':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel_7x7 = np.ones((7, 7), np.float32) / 49
            img_kernelled7x7 = cv2.filter2D(image, -1, kernel_7x7)
            img_kernelled7x7 = cv2.resize(img_kernelled7x7, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('Kernel 7x7 Blurring', img_kernelled7x7)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_kernelled7x7)
    if tkvar1.get() == 'Sobel CV_8U':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            sobelx8u = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
            sobelx8u = cv2.resize(sobelx8u, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('Sobel CV_8U', sobelx8u)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(sobelx8u)
    if tkvar1.get() == 'Sobel CV_64F':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            sobelx64f = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            abs_sobel64f = np.absolute(sobelx64f)
            sobel_8u = np.uint8(abs_sobel64f)
            sobel_8u = cv2.resize(sobel_8u, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('Sobel CV_64F', sobel_8u)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(sobel_8u)
    if tkvar1.get() == 'Edge Preserving Filter':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            img = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
            image = cv2.resize(image, (600, 500))
            img = cv2.resize(img, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('Edge Preserving Filter', shift_filter(img))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(shift_filter(img))
# Dropdown choices for manipulation
def do_manip():
    if tkvar2.get() == 'Resizing':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            if part_text3.get() !='' and part_text4.get() != '':
                r = int(part_text3.get()) / image.shape[1]
                dim = (int(part_text4.get()), int(image.shape[0] * r))
            else:
                r = 500 / image.shape[1]
                dim = (500, int(image.shape[0] * r))
            img_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('Original Image', image)
            cv2.imshow("Resized Image", img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_resized)
    if tkvar2.get() == 'Rotation':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            num_rows, num_cols = image.shape[:2]
            if part_text5.get() != '':
                rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), int(part_text5.get()), 1)
            else:
                rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 180, 1)
            img_rotated = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
            cv2.imshow('Original Image', image)
            cv2.imshow("Rotated Image", img_rotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_rotated)
    if tkvar2.get() == 'Cropping':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            if part_text6.get() != '' and part_text7.get() != '' and part_text8.get() != '' and part_text9.get() != '':
                img_cropped = image[int(part_text6.get()):int(part_text7.get()),
                              int(part_text8.get()):int(part_text9.get())]
            else:
                img_cropped = image[0:400, 0:500]
            cv2.imshow('Original Image', image)
            cv2.imshow("Cropped Image", img_cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_cropped)
    if tkvar2.get() == 'Perspective Transformation':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
            pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(image, M, (300, 300))
            image = cv2.resize(image, (600, 500))
            dst = cv2.resize(dst, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Perspective Transformation", dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(dst)
    if tkvar2.get() == 'Affine Transformation':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            rows, cols, ch = image.shape
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(image, M, (cols, rows))
            image = cv2.resize(image, (600, 500))
            dst = cv2.resize(dst, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Affine Transformation", dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(dst)
# Dropdown choices for intensity transformation
def do_intens():
    if tkvar3.get() == 'Image Negatives':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            img_negative = 255 - image
            image = cv2.resize(image, (600, 500))
            img_negative = cv2.resize(img_negative, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Image Negatives", img_negative)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_negative)
    if tkvar3.get() == 'Log Transformation':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            c = 255 / (np.log(1 + np.max(image)))
            img_log_transformed = c * np.log(1 + image)
            img_log_transformed = np.array(img_log_transformed, dtype=np.uint8)
            image = cv2.resize(image, (600, 500))
            img_log_transformed = cv2.resize(img_log_transformed, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Log Transformation", img_log_transformed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_log_transformed)
    if tkvar3.get() == 'Gamma Transformation':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            if part_text_gamma.get() != '':
                gamma_image = np.array(255 * (image / 255) ** int(part_text_gamma.get()), dtype='uint8')
            else:
                gamma_image = np.array(255 * (image / 255) ** 2.5, dtype='uint8')
            image = cv2.resize(image, (600, 500))
            gamma_image = cv2.resize(gamma_image, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Gamma Transformation", gamma_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(gamma_image)
    if tkvar3.get() == 'Piecewise Transformation':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            r1 = 70
            s1 = 0
            r2 = 140
            s2 = 255
            pixelVal_vec = np.vectorize(pixelval)
            image_last = pixelVal_vec(image, r1, s1, r2, s2)
            image = cv2.resize(image, (600, 500))
            image_last = cv2.resize(image_last, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Piecewise Transformation", image_last)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(image_last)
# Dropdown choices for morphological transformation
def do_morph():
    if tkvar4.get() == 'Erosion':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((5, 5), np.uint8)
            img_erosion = cv2.erode(image, kernel, iterations=1)
            image = cv2.resize(image, (600, 500))
            img_erosion = cv2.resize(img_erosion, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Erosion", img_erosion)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_erosion)
    if tkvar4.get() == 'Dilation':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.dilate(image, kernel, iterations=1)
            image = cv2.resize(image, (600, 500))
            img_dilation = cv2.resize(img_dilation, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Dilation", img_dilation)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_dilation)
    if tkvar4.get() == 'Opening':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((5, 5), np.uint8)
            img_opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            image = cv2.resize(image, (600, 500))
            img_opening = cv2.resize(img_opening, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Opening", img_opening)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_opening)
    if tkvar4.get() == 'Closing':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((5, 5), np.uint8)
            img_closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = cv2.resize(image, (600, 500))
            img_closing = cv2.resize(img_closing, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Closing", img_closing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_closing)
    if tkvar4.get() == 'Gradient':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((5, 5), np.uint8)
            img_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
            image = cv2.resize(image, (600, 500))
            img_gradient = cv2.resize(img_gradient, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Gradient", img_gradient)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_gradient)
    if tkvar4.get() == 'Top Hat':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((12, 12), np.uint8)
            img_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            image = cv2.resize(image, (600, 500))
            img_tophat = cv2.resize(img_tophat, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Top Hat", img_tophat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_tophat)
    if tkvar4.get() == 'Black Hat':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((12, 12), np.uint8)
            img_blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            image = cv2.resize(image, (600, 500))
            img_blackhat = cv2.resize(img_blackhat, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Black Hat", img_blackhat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_blackhat)
    if tkvar4.get() == 'Crossing':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((5, 5), np.uint8)
            img_crossing = cv2.morphologyEx(image, cv2.MORPH_CROSS, kernel)
            image = cv2.resize(image, (550, 500))
            img_crossing = cv2.resize(img_crossing, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Crossing", img_crossing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_crossing)
    if tkvar4.get() == 'Ellipse':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((10, 10), np.uint8)
            img_ellipse = cv2.morphologyEx(image, cv2.MORPH_ELLIPSE, kernel)
            image = cv2.resize(image, (550, 500))
            img_ellipse = cv2.resize(img_ellipse, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Ellipse", img_ellipse)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_ellipse)
    if tkvar4.get() == 'Rectangular':
        if part_text1.get() != '':
            image = cv2.imread(part_text1.get())
            kernel = np.ones((10, 10), np.uint8)
            img_rectangular = cv2.morphologyEx(image, cv2.MORPH_RECT, kernel)
            image = cv2.resize(image, (600, 500))
            img_rectangular = cv2.resize(img_rectangular, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Ellipse", img_rectangular)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_rectangular)
# Headings of dropdown lists
popupMenu1 = OptionMenu(app, tkvar1, *choices1)
label1 = Label(app, text="Choose a Filter",font="Times, 11",fg='blue')
label1.place(x=41, y=225)
popupMenu1.place(x=43, y=250)
my_button1 = Button(app, text='Activate', command=do_filter)
my_button1.place(x=60, y=290)
popupMenu2 = OptionMenu(app, tkvar2, *choices2)
label2 = Label(app, text="Choose a Manipulation Method", font="Times, 11",fg='blue')
label2.place(x=190, y=225)
popupMenu2.place(x=210, y=250)
my_button2 = Button(app, text='Activate', command=do_manip)
my_button2.place(x=254, y=290)
popupMenu3 = OptionMenu(app, tkvar3, *choices3)
label3 = Label(app, text="Choose an Intensity Transformation", font="Times, 11", fg='blue')
label3.place(x=430, y=225)
popupMenu3.place(x=445, y=250)
my_button3 = Button(app, text='Activate', command=do_intens)
my_button3.place(x=505, y=290)
popupMenu4 = OptionMenu(app, tkvar4, *choices4)
label4 = Label(app, text="Choose a Morphological Transformation", font="Times, 11",fg='blue')
label4.place(x=700, y=225)
popupMenu4.place(x=720, y=250)
my_button4 = Button(app, text='Activate', command=do_morph)
my_button4.place(x=795, y=290)
# Video editing function (gamma transformation)
def edit_video():
    # Video Sharpening with using Gamma Transformation
    if part_text2.get() != '':
        cap = cv2.VideoCapture(part_text2.get())
        while True:
            ret, frame = cap.read()
            frame1 = np.array(255 * (frame / 255) ** 1.7, dtype='uint8')
            frame2 = np.array(255 * (frame / 255) ** 2.7, dtype='uint8')
            cv2.imshow('Original Video', frame)
            cv2.imshow('Gamma(1.7) Transformation', frame1)
            cv2.imshow('Gamma(2.7) Transformation', frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
# Labels and button for video editing
labelvideo1 = Label(app, text="--> Video Sharpening with using Gamma Transformation <-- ", font="Times, 13", fg='green')
labelvideo1.place(x=450, y=360)
labelvideo2 = Label(app, text="(Hint: Press 'q' to close all videos!)", font="Times, 10")
labelvideo2.place(x=545, y=385)
video_button4 = Button(app, text='Edit Video', font="Times, 10", command=edit_video)
video_button4.place(x=610, y=415)
# Histogram equalization function
def histogram_equ():
    if part_text1.get() != '':
        img = cv2.imread(part_text1.get(),0)
        equ = cv2.equalizeHist(img)
        res = np.hstack((img, equ))  # stacking images side-by-side
        cv2.imshow('Histogram-Image', res)
        cv2.waitKey(0)
        save_image(res)
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.plot(cdf_normalized, color='b')
        plt.hist(img.flatten(), 256, [0, 256], color='black')
        plt.xlim([0, 256])
        plt.legend(('Cdf', 'Histogram'), loc='upper left')
        plt.show()
# Labels and button for histogram equalization
labelhistogram = Label(app, text="--> Histogram Equalization for Image <--", font="Times, 13", fg='green')
labelhistogram.place(x=80, y=360)
labelvideo2 = Label(app, text="(Hint: Enter the image name before activating)", font="Times, 10")
labelvideo2.place(x=85, y=385)
histogram_button = Button(app, text='Activate', font="Times, 10", command=histogram_equ)
histogram_button.place(x=185, y=415)
#Start Program
app.mainloop()

# ----Sergen KARATAŞ / 0515000637
