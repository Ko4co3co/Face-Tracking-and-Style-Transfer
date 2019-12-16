import tensorflow as tf 
from net import Tensorflow_model
from label_parser import v4_label_parser
import numpy as np
import glob 
import cv2 

class Infernece():
    def __init__(self,style_dir='./style_image',video_dir='./video'):
        self.style_dir = style_dir
        self.video_dir = video_dir

    def Style_Image_Load(self):
        image_lists = glob.glob(self.style_dir + '/*')
        if len(image_lists) == 0:
            raise AttributeError('Empty Style Images')
        style_images = [cv2.imread(path) for path in image_lists]        
        return style_images 
    
    def draw_bbox(self,img,result,label):
            SCORE_THRESHOLD = 0.5
            width , height , _ = img.shape 
            bboxes= result['detection_boxes']
            class_names = result['detection_classes']
            scores = result['detection_scores']
            draw_boxes = []
            for idx in range(len(scores)):
                if float(scores[idx].numpy()) >=SCORE_THRESHOLD:
                    draw_boxes.append([str(int(class_names[idx].numpy())),[int(bboxes[idx][0]*width),int(bboxes[idx][1]*height),int(bboxes[idx][2]*width),int(bboxes[idx][3]*height)]])
            if len(draw_boxes) >= 1:
                for i in range(len(draw_boxes)):
                    print(draw_boxes)
                    class_name = label[draw_boxes[i][0]]
                    ymin = draw_boxes[i][1][0]
                    xmin = draw_boxes[i][1][1]
                    ymax = draw_boxes[i][1][2]
                    xmax = draw_boxes[i][1][3]
                    img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color=(255,255,255),thickness=1)
                    img = cv2.putText(img,class_name,(xmin,ymin-10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(255,255,255),fontScale=1.0)
    
    def Video_Inference(self):
        detection_model_path = './model/object_detection/saved_model'        
        style_transfer_path = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        label = v4_label_parser().load_label()
        video = glob.glob(self.video_dir+'/*')          
        style_images = self.Style_Image_Load()
        style_image = np.expand_dims(style_images[0].astype(np.float32)/255.,axis=0)
        idx =0
        detection = Tensorflow_model(detection_model_path).load_weight_model()
        transfer_model = Tensorflow_model(style_transfer_path).load_hub_model()
        cap = cv2.VideoCapture(video[0])
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('./output/output.avi',fourcc,25.0,(int(cap.get(3)),int(cap.get(4))))
        while True:
            ret, frame = cap.read()
            if ret:
                height ,width ,_ = frame.shape
                matrix = cv2.getRotationMatrix2D((width/2, height/2), 270, 1)
                frame = cv2.warpAffine(frame, matrix, (width, height))
                detection_result = detection(tf.expand_dims(frame,axis=0))
                category = detection_result['detection_classes'][0]
                bbox = detection_result['detection_boxes'][0]
                for idx in range(len(category)):
                    #502 == Human Face
                    if category[idx].numpy() == 502.0:
                        width , height , _ = frame.shape
                        xmin = bbox[idx][0] * width
                        ymin = bbox[idx][1] * height
                        xmax = bbox[idx][2] * width
                        ymax = bbox[idx][3] * height
                        img = np.copy(frame[xmin:xmax,ymin:ymax])   
                        face_h,face_w ,_ =  img.shape                   
                        img = tf.expand_dims(img.astype(np.float32),axis=0)                                            
                        transfered = transfer_model(tf.constant(img),tf.constant(style_image))                              
                        transfered = cv2.resize(np.squeeze(transfered[0])*255,(face_w,face_h))
                        frame[xmin:xmax,ymin:ymax] = transfered
                
                out.write(frame)
                cv2.imshow('video', frame)
                cv2.waitKey(1)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()	  

if __name__ == "__main__":
    inf = Infernece()
    inf.Video_Inference()
 