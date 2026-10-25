import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')
labels = learn.dls.vocab

def classify(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(labels, map(float, probs)))

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
)
demo.launch()
