###           Written and Executed By:            ####
###               Sheekar Banerjee                ####
###             AI Engineering Lead               ####



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np


# Load saved model
model = load_model("trans-alzh-Model.h5")

# Get the model summary from the loaded model
print(model.summary())

# Input image
img_path = "C:/Users/SHEEKAR/PycharmProjects/ML Dense/Alzheimers Dataset/test/MildDemented/26 (25).jpg"

# Input target size 224 by 224 for loaded input image
img = image.load_img(img_path, target_size=(224, 224))

# Converting image to array
x = image.img_to_array(img)

# expanding the dimension
x = np.expand_dims(x, axis=0)

# preprocessing the input image under the dimension
x = preprocess_input(x)

# get prediction from model
preds=model.predict(x)

# Create a list containing the class labels
class_labels = ["Mild Dementia","Moderate Dementia","Non Dementia", "Very Mild Dementia"]

# Find the index of the class with maximum score
pred = np.argmax(preds, axis=-1)

# Print the label of the class with maximum score
print("\n The prediction is: ", class_labels[pred[0]])
