import cv2
import os
import string

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

for i in range(10):
    if not os.path.exists("data/train/" + str(i)):
        os.makedirs("data/train/" + str(i))
    if not os.path.exists("data/test/" + str(i)):
        os.makedirs("data/test/" + str(i))

for i in string.ascii_uppercase:
    if not os.path.exists("data/train/" + i):
        os.makedirs("data/train/" + i)
    if not os.path.exists("data/test/" + i):
        os.makedirs("data/test/" + i)

# Train or test 
mode = 'train'
directory = 'data/' + mode + '/'
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1  

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {}
    for i in range(10):
        count[str(i)] = len(os.listdir(directory + "/" + str(i)))
    for i in string.ascii_uppercase:
        count[i] = len(os.listdir(directory + "/" + i))
    
    # Printing the count in each set to the screen
    for i in range(10):
        cv2.putText(frame, f"{i} : {count[str(i)]}", (10, 70 + i * 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    for i in string.ascii_uppercase:
        cv2.putText(frame, f"{i} : {count[i]}", (10, 70 + (ord(i) - ord('A') + 10) * 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    
    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    
    # Drawing the ROI
    cv2.rectangle(frame, (x1 - 1, y1), (x2 + 1, y2), (255, 0, 0), 1)
    
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    cv2.imshow("Frame", frame)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("Gaussian Blur", test_image)
        
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF in range(ord('0'), ord('9') + 1):
        digit = chr(interrupt & 0xFF)
        cv2.imwrite(directory + digit + '/' + str(count[digit]) + '.jpg', roi)
    if interrupt & 0xFF in range(ord('A'), ord('Z') + 1):
        letter = chr(interrupt & 0xFF)
        cv2.imwrite(directory + letter + '/' + str(count[letter]) + '.jpg', roi)

cap.release()
cv2.destroyAllWindows()
