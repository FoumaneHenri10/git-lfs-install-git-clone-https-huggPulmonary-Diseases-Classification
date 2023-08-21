import gradio as gr
#import joblib
import tensorflow as tf

#model = joblib.load('model.pkl')
model = tf.keras.models.load_model('best_model.h5')
categories = ["Normal", "Tubercolosis", "Pneumonia"]

def classify(img):
    img = img.reshape((-1, 224, 224, 3))
    pred = model.predict(img)[0]
    return {categories[i]: float(pred[i]) for i in range(3)}

image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=2)
examples = ["Normal.png", "Tuberculosis.png", "Pneumonia.jpeg"]
    

intf = gr.Interface(classify,inputs=image, outputs=label, examples=examples, capture_session=True)
intf.launch(inline=False)
