from spellchecker  import SpellChecker
from keras.models import model_from_json
import cv2
import operator
from string import ascii_uppercase
from spellchecker import SpellChecker

class SignLanguageRecognizer:
    def __init__(self):
        self.directory = 'model/'
        self.loaded_models = {}
        self.load_models()
        self.spell_checker = SpellChecker()

    def recognize_sign_language(self, frame):
        # Preprocess the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Resize the frame to match the input size of the CNN model
        resized_frame = cv2.resize(res, (128, 128))
        
        # Predict the hand sign using the loaded CNN models
        predictions = {}
        for model_name, loaded_model in self.loaded_models.items():
            result = loaded_model.predict(resized_frame.reshape(1, 128, 128, 1))
            predictions[model_name] = result
        
        # Post-process the predictions to get the recognized symbols
        current_symbol = ''
        for model_name, result in predictions.items():
            prediction = {}
            prediction['blank'] = result[0][0]
            inde = 1
            for i in ascii_uppercase:
                prediction[i] = result[0][inde]
                inde += 1
            # Sort the predictions
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            current_symbol += prediction[0][0]
        
        # Suggest alternative words using a spell checker
        word_suggestions = self.spell_checker.correction(current_symbol)

        return current_symbol, word_suggestions

