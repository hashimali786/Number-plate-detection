# Number-plate-detection

# 1. Imports and Setup
Roboflow is used to load a pre-trained number plate detection model.
OpenCV (cv2) handles image processing.
EasyOCR is used for Optical Character Recognition (OCR).
Matplotlib is used to visualize images.
Warnings are ignored using warnings.filterwarnings("ignore").
# 2. Roboflow Model Loading (Sensitive Part)
python
Copy
Edit
rf = Roboflow(api_key="")  # API Key (Sensitive!)
project = rf.workspace().project("")  # Project Name (Sensitive!)
model = project.version(2).model
🔹 Privacy Concern: The API key and project details should not be public. Instead:

Store the API key in an environment variable (os.environ["ROBOFLOW_API_KEY"]).
Load it securely instead of hardcoding.
# 3. Image Loading and Preprocessing
python
Copy
Edit
img_path = 'data/014.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Reads the image from disk and converts it to RGB format.
# 4. Running Object Detection
python
Copy
Edit
results = model.predict(img_rgb, confidence=50, overlap=30).json()
print("results", results)
The model detects number plates with 50% confidence threshold.
Predictions are returned in JSON format.
# 5. Processing Detected Number Plates
python
Copy
Edit
for prediction in results['predictions']:
    class_name = prediction['class']
    if class_name == '24':  # Custom class name condition
        print("Detected a number plate")
Loops through detected objects.
Filters based on class name (24 is used for number plates in this case).
# 6. Extracting and Enhancing Number Plate Image
python
Copy
Edit
x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
license_plate = img[int(y - (height / 2)):int(y + (height / 2)), int(x - (width / 2)):int(x + (width / 2)), :].copy()
Extracts the detected number plate using bounding box coordinates.
Applies denoising, grayscale conversion, and morphological operations to enhance text clarity.
# 7. Optical Character Recognition (OCR)
python
Copy
Edit
reader = easyocr.Reader(['en'])
output = reader.readtext(license_plate_gray)
for out in output:
    text, text_score = out[-2], out[-1]
    if text_score > 0.4:
        print("Extracted Text:", text, "Confidence:", text_score)
Uses EasyOCR to extract text from the cleaned number plate.
Filters results based on confidence score (> 0.4).
# 8. Visualization
python
Copy
Edit
plt.figure()
plt.title("license_plate_gray")
plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))
plt.show()
Displays processed images for debugging.
python
Copy
Edit
import cv2
blurred = cv2.GaussianBlur(license_plate_gray, (25, 25), 0)
plt.imshow(blurred, cmap="gray")
# Final Thoughts
The script detects, extracts, and reads vehicle number plates.
It leverages AI via Roboflow for detection and EasyOCR for recognition.
