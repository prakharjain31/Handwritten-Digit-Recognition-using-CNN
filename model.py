import tensorflow as tf
from PIL import Image
import numpy as np
import io


new_model = tf.keras.models.load_model('digits_recognition_cnn.keras')

def model_predict(img):
    input_data = img.astype(np.float32)

    y = new_model.predict(input_data)
    return y.argmax(), y[0][y.argmax()]
 
def preprocess_image(image, img_type="file"):
    if img_type == "file":
        return (image.reshape(1, 28, 28, 1) / 255.).astype(np.float32)
    elif img_type == "df_row":
        return (image.to_numpy().reshape(1, 28, 28, 1) / 255.).astype(np.float32)

def predict(image_data):
    image = Image.open(io.BytesIO(image_data))

    image = image.convert("L")

    image = image.resize((28, 28))

    image = np.array(image)

    image = image.reshape(1, 28, 28, 1) 

    image = image / 255.

    image = image.astype(np.float32)

    prediction, confidence = model_predict(image)

    return prediction, confidence
    
    