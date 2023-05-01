### **Face Recognition Using MTCNN and FaceNet** 

- Used **MTCNN** (Multi-Task Cascaded Convolutional Neural Networks) to detect faces and facial landmarks on images.
- Extracted features from face images by **FaceNet** and outputted the 128-dimensional vector embedding for modeling.
- Calculated **Cosine similarity** between vector embeddings and determined if they are the same person.
- Used AFD (Asian Face Image Dataset) to implement **fine-tuning** in the training results based on CASIA-WebFace dataset.
- Greatly Improved the accuracy of the model on Asian faces from **89.8%** to **99.7%**.
