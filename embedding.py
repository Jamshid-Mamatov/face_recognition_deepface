import torch
import torch.nn.functional as F

from pathlib import Path
from facenet_pytorch import MTCNN,InceptionResnetV1
import cv2
import json
from detector import detect_face
import torchvision.transforms as transforms
from PIL import Image
mtcnn=MTCNN()

resnet = InceptionResnetV1(pretrained='vggface2').eval()

# paths=list(Path("data_team").glob("*"))
# item={}
# for path in paths:

    
#     name=path.name[:-4]
#     img1=cv2.imread(path.as_posix())
#     img_cropped1=mtcnn(img1)

#     img_embedding = resnet(img_cropped1.unsqueeze(0))
 
#     item[name]=img_embedding
#     print(name)
# torch.save(item, "data_team.json")
# loaded=torch.load("data_team.json")

# print(loaded['Jamshid Mamatov'])

img=cv2.imread("team.jpg")
faces=detect_face(img)
data=torch.load("data_team.json")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to fit the model's input size
    transforms.ToTensor(),  # Convert to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

for face in faces:
    
    

    # Load and preprocess an image
   
    image = Image.fromarray(face)
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0) 

    img_embedding= resnet(img_tensor)
    
    for member in data:
        print(member)
        cosine_similarity_score = F.cosine_similarity(img_embedding, data[member]).item()


        print("Cosine Similarity between Image Embeddings:", cosine_similarity_score*100,"name:",member)
        if cosine_similarity_score*100>15:
            

    print("\n\n\n")