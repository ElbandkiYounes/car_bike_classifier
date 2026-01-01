import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Charger le modèle
model = tf.keras.models.load_model("car_bike_model.h5")

# Charger une image
img_path = "test.jpg"  # mets ici ton image
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Prédiction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("🟢 Moto")
else:
    print("🔵 Voiture")
