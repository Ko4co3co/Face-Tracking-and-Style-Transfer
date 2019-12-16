import tensorflow as tf 
import tensorflow_hub as hub 
import cv2 
import glob
class Tensorflow_model():
    def __init__(self,model_path=None):
        # For Object Detection is Local path
        # For Style Transfer is TF HUB Path
        self.model_path = model_path
        
    
    def load_hub_model(self):
        return hub.load(self.model_path)
    
    def load_weight_model(self):
        model = tf.saved_model.load(self.model_path)
        model = model.signatures['serving_default']
        return model
    


if __name__ == "__main__":
    style = Tensorflow_model('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    style = style.load_hub_model()
    print(style)
    ob = Tensorflow_model('./model/object_detection/saved_model')
    ob = ob.load_weight_model()
    print(ob)