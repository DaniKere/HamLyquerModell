import model as md
import tensorflow as tf

model = md.create_model()
model.load_weights(md.WEIGHTS_PATH)


