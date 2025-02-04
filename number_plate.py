from roboflow import Roboflow
import cv2
import easyocr
import matplotlib.pyplot as plt
import warnings
# import pytesseract
# from PIL import Image
# import keras_ocr  # For Keras-OCR

# Ignore specific warning (replace WarningType with the actual warning type)
warnings.filterwarnings("ignore")

# Path to Tesseract executable (if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Code that generates the warning
rf = Roboflow(api_key="")
project = rf.workspace().project("npd")
model = project.version(2).model

# rf = Roboflow(api_key="")
# project = rf.workspace().project("number-plates")
# model = project.version(1).model

# Load and process the image
img_path = 'data/014.jpg'
img = cv2.imread(img_path)
# 1672075528491 readed with denoicing
# # Define non-integer coordinates (x, y)
# x = 50
# y = 100
#
# # Perform bilinear interpolation
# interpolated_value = cv2.getRectSubPix(img, (2200, 900), (x, y))
# # Resulting interpolated_value contains the interpolated pixel value
# print(interpolated_value)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_rgb = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)


# Infer using the model
results = model.predict(img_rgb, confidence=50, overlap=30).json()
print("results", results)

# Loop through each prediction
for prediction in results['predictions']:
    class_name = prediction['class']
    # if class_name == 'number plate':
    # if class_name == 'Plates':
    if class_name == '24':
        print("Detected a number plate")
        # for detect in results['predictions']:
        #     number_plate = []
        #     if class_name == '0':
        #         number_plate.append(0)
        #     elif class_name == '1':
        #         number_plate.append(1)
        #     elif class_name =='2':
        #         number_plate.append(class_name)
        #     elif class_name =='3':
        #         number_plate.append(class_name)
        #     elif class_name =='4':
        #         number_plate.append(class_name)
        #     elif class_name == '5':
        #         number_plate.append(class_name)
        #     elif class_name == '6':
        #         number_plate.append(class_name)
        #     elif class_name == '7':
        #         number_plate.append(class_name)
        #     elif detect == '8':
        #         number_plate.append(8)
        #     elif detect == '9':
        #         number_plate.append(9)
        #     elif class_name == '10':
        #         class_name = "A"
        #         number_plate.append(class_name)
        #     elif class_name == '11':
        #         class_name = "B"
        #         number_plate.append(class_name)
        #     elif class_name == '12':
        #         class_name = "C"
        #         number_plate.append(class_name)
        #     elif class_name == '13':
        #         class_name = "D"
        #         number_plate.append(class_name)
        #     elif class_name == '14':
        #         class_name = "E"
        #         number_plate.append(class_name)
        #     elif class_name == '15':
        #         class_name = "F"
        #         number_plate.append(class_name)
        #     elif class_name == '16':
        #         class_name = "G"
        #         number_plate.append(class_name)
        #     elif class_name == '17':
        #         class_name = "H"
        #         number_plate.append(class_name)
        #     elif class_name == '18':
        #         class_name = "I"
        #         number_plate.append(class_name)
        #     elif class_name == '19':
        #         class_name = "J"
        #         number_plate.append(class_name)
        #     elif class_name == '20':
        #         class_name = "K"
        #         number_plate.append(class_name)
        #     elif class_name == '21':
        #         class_name = "L"
        #         number_plate.append(class_name)
        #     elif class_name == '22':
        #         class_name = "N"
        #         number_plate.append(class_name)
        #     elif class_name == '23':
        #         class_name = "O"
        #         number_plate.append(class_name)
        #     elif class_name == '25':
        #         class_name = "R"
        #         number_plate.append(class_name)
        #     elif class_name == '26':
        #         class_name = "S"
        #         number_plate.append(class_name)
        #     elif class_name == '27':
        #         class_name = "T"
        #         number_plate.append(class_name)
        #     elif class_name == '28':
        #         class_name = "U"
        #         number_plate.append(class_name)
        #     elif class_name == '29':
        #         class_name = "V"
        #         number_plate.append(class_name)
        #     elif class_name == '30':
        #         class_name = "X"
        #         number_plate.append(class_name)
        #     elif class_name == '31':
        #         class_name = "Z"
        #         number_plate.append(class_name)

# print("number_plate", number_plate)

        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        print("Bounding Box Coordinates:", x, y, width, height)

        # Extract the region of the image corresponding to the detected number plate
        license_plate_img = img[int(y):int(y + height), int(x):int(x + width)]

        license_plate = img[int(y - (height / 2)):int(y + (height / 2)), int(x - (width / 2)):int(x + (width / 2)), :].copy()

        img = cv2.rectangle(img,
                            (int(x - (width / 2)), int(y - (height / 2))),
                            (int(x + (width / 2)), int(y + (height / 2))),
                            (0, 255, 0),
                            15)
        # Apply denoising
        denoised_license_plate = cv2.fastNlMeansDenoising(license_plate, None, h=10, templateWindowSize=7,
                                                          searchWindowSize=21)


        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY)

        # Apply thresholding to create a binary image
        _, binary_plate = cv2.threshold(license_plate_gray, 128, 255, cv2.THRESH_BINARY)

        # Define a horizontal kernel (adjust the kernel size as needed)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

        # Apply morphology to remove horizontal lines
        removed_horizontal_lines = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, horizontal_kernel)

        # Define a vertical kernel (adjust the kernel size as needed)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

        # Apply morphology to remove vertical lines
        removed_vertical_lines = cv2.morphologyEx(removed_horizontal_lines, cv2.MORPH_OPEN, vertical_kernel)

        plt.figure()
        plt.title("removed_vertical_lines")
        plt.imshow(cv2.cvtColor(removed_vertical_lines, cv2.COLOR_BGR2RGB))
        plt.show()

        # Apply Gaussian blur for noise reduction
        blurred_plate = cv2.GaussianBlur(removed_vertical_lines, (5, 5), 0)

        # Enhance contrast
        # license_plate_gray = cv2.equalizeHist(grayy)

        # _, license_plate_thresh = cv2.threshold(
        #     license_plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        # )
        # Apply thresholding
        _, license_plate_thresh = cv2.threshold(blurred_plate, 148, 255, cv2.THRESH_BINARY_INV)

        # Use pytesseract to perform OCR on the image
        # output = pytesseract.image_to_string(license_plate_thresh)
        # Specify language (e.g., English)
        # output = pytesseract.image_to_string(license_plate_gray, lang='eng')
        # Print the extracted text
        # print("Extracted Text:", output)

        # Perform OCR using EasyOCR
        reader = easyocr.Reader(['en'])
        output = reader.readtext(license_plate_gray)


        for out in output:
            text, text_score = out[-2], out[-1]
            if text_score > 0.4:
                print("Extracted Text:", text, "Confidence:", text_score)
        plt.figure()
        plt.title("img 1")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        # plt.figure()
        # plt.title("license plate")
        # plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
        # plt.show()

        plt.figure()
        plt.title("license_plate_gray")
        plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))
        plt.show()

        # plt.figure()
        # plt.title("license_plate_thresh")
        # plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))
        # plt.show()

        # # Visualize the detected license plate
        # plt.figure()
        # plt.imshow(cleaned_image)
        # plt.title("Detected License Plate")
        # # plt.axis('off')
        # plt.show()
