from django.shortcuts import render
from django.http import HttpResponse 
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_POST
from .sign_language_recognizer import SignLanguageRecognizer
import tensorflow as tf
import numpy as np
import cv2
import base64
import json
from keras.models import model_from_json
from string import ascii_uppercase
import operator
from spellchecker import SpellChecker
import os
import uuid
from PIL import Image
from datetime import datetime
# Create your views here.

def home(request):
    return render(request, 'home.html')

def about(request):
    # return HttpResponse('This is about page')
    return render(request, 'about.html')

def register(request):
    return render(request, 'register.html')

def login(request):
    return render(request, 'login.html')

# Define the SignLanguageRecognizer class

class SignLanguageRecognizer:
    def __init__(self):
        self.directory = 'D:/Final Year project/Project/unitalk/student/model'
        self.spell_checker = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.str = ""
        self.blank_flag = 0
        self.word = ""
        
        # Load models
        self.loaded_model = self.load_model("atoz")
        self.loaded_model_dru = self.load_model("model-bw_dru")
        self.loaded_model_tkdi = self.load_model("model-bw_tkdi")
        self.loaded_model_smn = self.load_model("model-bw_smn")
        
        # Initialize letter count
        self.ct = {'blank': 0}
        for letter in ascii_uppercase:
            self.ct[letter] = 0

        print("Loaded models from disk")

    def load_model(self, model_name):
        json_file = open(os.path.join(self.directory, f"{model_name}.json"), "r")
        model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(os.path.join(self.directory, f"{model_name}.h5"))
        return loaded_model

    def process_frames(self, frame):
        print("Processing frames...")
        cv2image = cv2.flip(frame, 1)
        x1 = int(0.5 * frame.shape[1])
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = int(0.5 * frame.shape[1])
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0), 1)
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
        self.current_image = Image.fromarray(cv2image)
        cv2image = cv2image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Resize the processed image to 128x128 pixels
        resized_image = cv2.resize(res, (128, 128))

        # Save the processed image using OpenCV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_image_filename = f"processed_image_{timestamp}.jpg"
        processed_image_dir = 'D:/Final Year project/Project/unitalk/student/images'
        os.makedirs(processed_image_dir, exist_ok=True)  # Ensure the directory exists or create it
        processed_image_path = os.path.join(processed_image_dir, processed_image_filename)
        cv2.imwrite(processed_image_path, resized_image)

        predicted_text = self.predict(resized_image)
        print("Predicted text:", predicted_text)
        return predicted_text

    def predict(self, test_image):
        print("Predicting...")
        print("Test image shape:", test_image.shape)
        test_image = cv2.resize(test_image, (128, 128))
        print("Test image shape:", test_image.shape)
        
        # Initialize current_symbol with a default value
        current_symbol = 'A'  # Choose a default symbol (could be any valid symbol)
        
        # Reset self.str to an empty string
        self.str = ""
        
        # Perform predictions using loaded models
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))
        
        # Concatenate all prediction results
        combined_result = np.concatenate((result, result_dru, result_tkdi, result_smn), axis=1)
        
        # Create a list of letters from 'A' to 'Z' and 'blank'
        letters = list(ascii_uppercase) + ['blank']
        
        # Initialize a dictionary to store predictions
        prediction = dict(zip(letters, combined_result[0]))
        
        # Sort predictions in descending order
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        
        # Get the most probable symbol
        current_symbol = prediction[0][0]
        
        # Update self.str whenever a valid symbol is obtained
        if current_symbol != 'blank':
            self.str = current_symbol
        
        # Handle blank symbol and update self.word
        if current_symbol == 'blank':
            # Reset letter counts
            for letter in ascii_uppercase:
                self.ct[letter] = 0
            self.ct['blank'] = 0
            
            # Process self.word if it's not empty
            if self.blank_flag == 0 and len(self.word) > 0:
                self.blank_flag = 1
                self.str += " " + self.word
                self.word = ""
        else:
            # Update self.word and reset blank_flag
            if len(self.str) > 16:
                self.str = ""
            self.blank_flag = 0
            self.word += current_symbol
            print("Updated self.word:", self.word)
        
        return self.str






def process_frame(request):
    if request.method == 'GET':
        return render(request, 'translate.html')

    elif request.method == 'POST':
        # Process the received frame data
        image_data = request.POST.get('image_data')

        # Convert base64 image data to numpy array
        decoded_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(decoded_data, np.uint8)
        print("Size of nparr:", nparr.shape)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Initialize SignLanguageRecognizer if not already initialized
        if 'recognizer' not in process_frame.__dict__:
            process_frame.recognizer = SignLanguageRecognizer()

        # Process the frame using SignLanguageRecognizer method
        recognized_text = process_frame.recognizer.process_frames(frame)
        # print("Final Output:", recognized_text)  # Print final output in the terminal

        return JsonResponse({'recognized_text': recognized_text})
    else:
        return JsonResponse({'error': 'Invalid request method'})





# def translate(request):
#     if request.method == 'GET':
#         return render(request, 'translate.html')
#     elif request.method == 'POST':
#         sign_language_recognizer = SignLanguageRecognizer()
#         predicted_text_generator = sign_language_recognizer.video_loop()
#         predicted_text = next(predicted_text_generator)
#         return JsonResponse({'text': predicted_text})



# class SignLanguageRecognizer:
#     def __init__(self):
#         self.directory = 'model/'
#         self.loaded_models = {}
#         self.load_models()
#         self.spell_checker = SpellChecker()

#     def load_models(self):
#         self.loaded_models['model'] = tf.keras.models.load_model(r'D:\Final Year project\Project\unitalk\model-bw.h5')

#     def preprocess_frame(self, frame):
#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Apply Gaussian blur
#         blur = cv2.GaussianBlur(gray, (5, 5), 2)
#         # Apply adaptive thresholding
#         th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#         # Apply Otsu's thresholding
#         _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         # Resize frame to match the input size of the CNN model
#         resized_frame = cv2.resize(res, (128, 128))
#         # Normalize frame
#         resized_frame = resized_frame / 255.0
#         return resized_frame

#     def recognize_sign_language(self, frame):
#         # Preprocess the frame
#         resized_frame = self.preprocess_frame(frame)
        
#         # Predict the hand sign using the loaded CNN models
#         predictions = {}
#         for model_name, loaded_model in self.loaded_models.items():
#             result = loaded_model.predict(resized_frame.reshape(1, 128, 128, 1))
#             predictions[model_name] = result
        
#         # Post-process the predictions to get the recognized symbols
#         current_symbol = ''
#         for model_name, result in predictions.items():
#             prediction = {}
#             prediction['blank'] = result[0][0]
#             inde = 1
#             for i in ascii_uppercase:
#                 prediction[i] = result[0][inde]
#                 inde += 1
#             # Sort the predictions
#             prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
#             current_symbol += prediction[0][0]
        
#         # Suggest alternative words using a spell checker
#         word_suggestions = self.spell_checker.correction(current_symbol)

#         return current_symbol, word_suggestions

#     def contains_hand_sign(self, frame):

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Apply thresholding to segment hand regions
#         _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Find contours in the thresholded image
#         contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Check if any contours are found
#         return len(contours) > 0


# # Initialize the SignLanguageRecognizer object
# sign_language_recognizer = SignLanguageRecognizer()

# # Instantiate SpellChecker
# spell_checker = SpellChecker()

# #prediction Check
# #1
# #####

# @require_http_methods(["GET", "POST"])
# def process_image(request):
#     if request.method == 'GET':
#         return render(request, 'recognize_sign_language.html')
#     elif request.method == 'POST':
#         # Decode and preprocess the posted image data
#         image_data = request.POST.get('image_data')
#         image_data = image_data.split(",")[1]  # Remove the data URI prefix
#         decoded_data = base64.b64decode(image_data)
#         np_data = np.frombuffer(decoded_data, dtype=np.uint8)
#         frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

#         # Check if the frame contains a hand sign
#         if sign_language_recognizer.contains_hand_sign(frame):
#             # Preprocess the frame
#             resized_frame = sign_language_recognizer.preprocess_frame(frame)

#             # Save the preprocessed image
#             save_path = os.path.join('preprocessed_images', f'preprocessed_image_{uuid.uuid4()}.jpg')
#             cv2.imwrite(save_path, resized_frame * 255)  # Save as JPEG with pixel values scaled back to 0-255 range

#             # Use SignLanguageRecognizer to recognize sign language
#             current_symbol, _ = sign_language_recognizer.recognize_sign_language(frame)

#             # Print the predicted letter or number to the terminal
#             print("Predicted Symbol:", current_symbol)

#             # Return the predicted symbol as a JSON response
#             return JsonResponse({'predicted_symbol': current_symbol})
#         else:
#             # If no hand sign is detected, return a blank response
#             return JsonResponse({'predicted_symbol': ''})
#     else:
#         # Return an error response for other types of requests
#         return JsonResponse({'error': 'Invalid Request'}, status=400)






#1
# @require_http_methods(["GET", "POST"])
# def process_image(request):
#     if request.method == 'GET':
#         return render(request, 'recognize_sign_language.html')
#     elif request.method == 'POST':
#         # Decode and preprocess the posted image data
#         image_data = request.POST.get('image_data')
#         image_data = image_data.split(",")[1]  # Remove the data URI prefix
#         decoded_data = base64.b64decode(image_data)
#         np_data = np.frombuffer(decoded_data, dtype=np.uint8)
#         frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

#         # Use SignLanguageRecognizer to recognize sign language
#         current_symbol, word_suggestions = sign_language_recognizer.recognize_sign_language(frame)

#         # Keep track of recognized symbols over multiple frames to form a sentence
#         # (You may need to implement more sophisticated logic depending on your requirements)
#         # For simplicity, let's assume each recognized symbol is a word
#         if 'sentence' not in request.session:
#             request.session['sentence'] = []
#         request.session['sentence'].append(current_symbol)

#         # Return the current sentence as a JSON response
#         current_sentence = ' '.join(request.session['sentence'])
#         return JsonResponse({'current_sentence': current_sentence, 'word_suggestions': word_suggestions})
#     else:
#         # Return an error response for other types of requests
#         return JsonResponse({'error': 'Invalid Request'}, status=400)


#3
# @require_http_methods(["GET", "POST"])
# def process_image(request):
#     if request.method == 'GET':
#         return render(request, 'recognize_sign_language.html')
#     elif request.method == 'POST':
#         try:
#             # Decode and preprocess the posted image data
#             image_data = request.POST.get('image_data')
#             image_data = image_data.split(",")[1]  # Remove the data URI prefix
#             decoded_data = base64.b64decode(image_data)
#             np_data = np.frombuffer(decoded_data, dtype=np.uint8)
#             frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            
#             # Use SignLanguageRecognizer to recognize sign language
#             current_symbol, word_suggestions = sign_language_recognizer.recognize_sign_language(frame)

#             # Keep track of recognized symbols over multiple frames to form a sentence
#             if 'sentence' not in request.session:
#                 request.session['sentence'] = []
#             request.session['sentence'].append(current_symbol)

#             # Return the current sentence as a JSON response
#             current_sentence = ' '.join(request.session['sentence'])
#             return JsonResponse({'current_sentence': current_sentence, 'word_suggestions': word_suggestions})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=400)
#     else:
#         # Return an error response for other types of requests
#         return JsonResponse({'error': 'Invalid Request'}, status=400)


#2
# @require_http_methods(["GET", "POST"])
# def process_image(request):
#     if request.method == 'GET':
#         return render(request, 'recognize_sign_language.html')
#     elif request.method == 'POST':
#         # Decode and preprocess the posted image data
#         image_data = request.POST.get('image_data')
#         image_data = image_data.split(",")[1]  # Remove the data URI prefix
#         decoded_data = base64.b64decode(image_data)
#         np_data = np.frombuffer(decoded_data, dtype=np.uint8)
#         frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        
#         # Convert frame to grayscale
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Preprocess the frame
#         resized_frame = sign_language_recognizer.preprocess_frame(gray_frame)

#         # Use SignLanguageRecognizer to recognize sign language
#         current_symbol, word_suggestions = sign_language_recognizer.recognize_sign_language(resized_frame)

#         # Return the predicted text as JSON response
#         return JsonResponse({'predicted_text': current_symbol, 'word_sequence': word_suggestions})
#     else:
#         # Return an error response for other types of requests
#         return JsonResponse({'error': 'Invalid Request'}, status=400)
