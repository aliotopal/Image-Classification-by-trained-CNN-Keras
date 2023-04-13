from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16


model = VGG16(weights='imagenet')  # Load the pretrained model (VGG16 trained on ImageNet dataset).

img = load_img("Cheetah.jpeg" , target_size=(224, 224))  # Load and resize the input image to CNN input size.

image = img_to_array(img)   # Convert it to array
image = image.reshape((1, 224, 224, 3)) # Reshape it for tensorflow to preprocess the image.
image = preprocess_input(image)  # Preprocess image for the VGG16
yhat = model.predict(image)     # Predict the image, it returns an array of 1x1000. Label values for each categories in the ImageNet
label0 = decode_predictions(yhat, top=5) # Reports top 5 predictions with label and label values
label1 = label0[0][0]  # Gets the Top1 label and values for reporting.
print("Image is classified as: %s  %.4f%%" % (label1[1], label1[2]))
print("Top 5 predictions are :", label0)
