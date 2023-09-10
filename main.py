from deepface import DeepFace



embedding_objs = DeepFace.represent(img_path = "detected_face_0.png")
embedding = embedding_objs[0]["embedding"]
print(embedding_objs)
