import cv2
import numpy as np
from facenet_pytorch import MTCNN,extract_face

import torch
import torch.nn.functional as F

from pathlib import Path
from facenet_pytorch import MTCNN,InceptionResnetV1

import torchvision.transforms as transforms
from PIL import Image,ImageDraw
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def detect_face(img):
    mtcnn=MTCNN()
    boxes,probs,points=mtcnn.detect(img,landmarks=True)

    img=np.array(img)
    image=img.copy()
    faces=[]
    data=torch.load("data_team.json")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to fit the model's input size
        transforms.ToTensor(),  # Convert to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
    ])
    
    for i,(box,prob,ld) in enumerate( zip(boxes,probs,points)):
        x,y,w,h=box
        # face=extract_face(image, box)
        # print(box,x,y,w,h)
        x=int(x)
        y=int(y)
        w=int(w)
        h=int(h)
        print(x,y,w,h)
        
        face=img[y:h,x:w]
        # cv2.imshow("frame",image[y:h,x:w])
        # cv2.imwrite("result.jpg",image[y:h,x:w])
        # if cv2.waitKey(1)==ord("q"):
        #     break
    

        
        

        # Load and preprocess an image
    
        face = Image.fromarray(face)
        img_tensor = preprocess(face)
        img_tensor = img_tensor.unsqueeze(0) 

        img_embedding= resnet(img_tensor)
        
        for member in data:
            print(member)
            cosine_similarity_score = F.cosine_similarity(img_embedding, data[member]).item()


            print("Cosine Similarity between Image Embeddings:", cosine_similarity_score*100,"name:",member)
            
            if cosine_similarity_score*100>30:
                
                # image1=ImageDraw.Draw(image1)
                # # image1.rectangle([(x, y), (w,h)])
                # image1.text((h,x),member,fill ="red", align ="right")
                cv2.rectangle(img, (x, y), (w,h), (0, 255, 0), 2)
                cv2.putText(img, member, (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        print("\n\n\n")
    # print(type(image1))
    # image.save("ravshan1.jpg")
    
    cv2.imwrite("result.jpg",img)

    return 0

img=cv2.imread("team.jpg")
faces=detect_face(img)