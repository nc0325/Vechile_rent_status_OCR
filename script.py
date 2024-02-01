import cv2
import pytesseract
from PIL import Image
import numpy as np
import io, json
from datetime import timedelta,datetime


# Path to tesseract executable
# On Windows, it might look like this: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Unix systems, just 'tesseract' is often sufficient if it's added to the PATH.
# pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR'


video_file = 'demo.mp4'

def validate_number(numeric_str):
    numeric_str_no_commas = numeric_str.replace(',', '').strip()

    if numeric_str_no_commas.isdigit() and int(numeric_str_no_commas):
        return numeric_str_no_commas
    else:
        # If not a valid number, return the default numeric value
        return None
    

def validate_lot(numeric_str):
    if validate_number(numeric_str) :
        if len(numeric_str) < 5:
            return None
        
        return numeric_str[-5:]
    else:
        # If not a valid number, return the default numeric value
        return None
   
        
        
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def get_sold_info(image):
    # Define the crop area (left, top, width, height)
    # Adjust these values for your specific cropping requirements
    h, w, c = image.shape
    x, y,= 280, 400
    crop_img = image[y:h, 0:x]
    # Convert image to RGB (OpenCV uses BGR by default)
    rgb_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    
    blur = cv2.GaussianBlur(rgb_image, (1,1), 0)

    inverted_rgb_image = cv2.bitwise_not(blur)

    # bright_text = increase_brightness(inverted_rgb_image, value=30)

    # Convert the isolated text image to grayscale
    gray_text = cv2.cvtColor(inverted_rgb_image, cv2.COLOR_RGB2GRAY)


    alpha = 3.1 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)

    constrasted = cv2.convertScaleAbs(gray_text, alpha=alpha, beta=beta)

    # Threshold the grayscale image to obtain a binary image
    

    # adapt_thresh = cv2.adaptiveThreshold(constrasted, 255,
    #                                     cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                     cv2.THRESH_BINARY_INV, 3, 2)

    texts = pytesseract.image_to_string(constrasted, lang='eng').split('\n')
    
    # print(texts)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('inverted_rgb_image', inverted_rgb_image)

    # cv2.imshow('gray_text Image', gray_text)

    # cv2.imshow('constrasted Image', constrasted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    lotIndex = -1
    for index, text in enumerate(texts):
        if(text.lower().find("sold") > -1):
            lotIndex = index
            break

    lotNumber = 0
    salePrice = 0
    if(lotIndex > -1 and lotIndex < len(texts) - 1):
        lotText = texts[lotIndex]
        for text in lotText.split(' '):
            if(validate_lot(text)):
                lotNumber = validate_lot(text)
                break
        bidText = ''
        if(texts[lotIndex].find('£') > -1) :
            bidText = texts[lotIndex]
        else : 
            bidText = texts[lotIndex + 1]
        
        salePrice = validate_number(bidText.split('£')[-1])


    if(lotNumber and salePrice):
        return {
            'lotNumber': lotNumber,
            'salePrice': salePrice
        }
    else:
        return None


def find_lot_index(sold_data, lot_text):
    last_index = -1
    for index, item in enumerate(sold_data):
        if item['lotNumber'] == lot_text :
                last_index = index
    
    return last_index
        
def filter_solddata(sold_data) :
    i = 0
    result = []
    while i < len(sold_data):
        current_item = sold_data[i]
        last_index = find_lot_index(sold_data, current_item['lotNumber'])
        last_item = sold_data[last_index]
        if(last_item['time'] - current_item['time'] > 1000):

            item = {
                'lotNumber': current_item['lotNumber'],
                'salePrice': current_item['salePrice'],
            }
            
            result.append(item)
        i = last_index + 1
    return result

processStartTime = datetime.today()
processStartSec = datetime.now().timestamp()
# Open the video file
cap = cv2.VideoCapture(video_file)

# Check if the video has been opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the interval between frames (every 3 seconds)
interval = int(fps * 3)

# Initialize frame number counter
frame_number = 0

soldData = []

while True:
    # Read a new frame
    ret, frame = cap.read()

    # Check if frame reading was successful
    if not ret:
        break  # Break the loop if we've reached the end of the video

    # Process only the frames that occur every 3 seconds
    if frame_number % interval == 0:

        soldInfo = get_sold_info(frame)
        
        print("sold info:" , soldInfo, "time: ", int(frame_number / 60 / fps) , ":" , int(frame_number / fps) % 60)

        if soldInfo:
            soldInfo['time'] = int(frame_number)
            soldData.append(soldInfo)

    # Increment the frame number
    frame_number += 1
    
# Release the VideoCapture object and close any open windows
cap.release()

filted_item_list = filter_solddata(soldData)

processDurationTime = timedelta(seconds= int(datetime.now().timestamp() - processStartSec))

result = {
    'videoSourceFile': video_file,
    "processStartTime": processStartTime.strftime('%d/%m/%Y %H:%M:%S'),
    'processDurationHH:MM:SS': str(processDurationTime),
    'salePrices': filted_item_list
}

print("result data:" , result)


with io.open('output.json', 'w', encoding='utf-8') as f:
  f.write(json.dumps(result, ensure_ascii=False))
  print(f'Object successfully saved ')


