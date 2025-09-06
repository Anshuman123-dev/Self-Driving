# 🛣️ Self-Driving Car Simulator using CNN

This project implements a **self-driving car simulator** using **Convolutional Neural Networks (CNNs)** for behavioral cloning. The model is trained to predict steering angles directly from images captured by the simulator, enabling end-to-end autonomous driving.

---

## 🚗 Project Overview
- Built using **Python, TensorFlow/Keras, and OpenCV** on **Google Colab**.  
- Trained on image data collected from the **Udacity Self-Driving Car Simulator**.  
- Inspired by and implemented based on the NVIDIA research paper:  
  *[End to End Learning for Self-Driving Cars (NVIDIA, 2016)](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)*.  
- The simulator provides center, left, and right camera images, along with steering angle data, which are used to train the network.  

---

## 📊 Model Architecture
The CNN architecture closely follows the NVIDIA model:
- Convolutional layers for feature extraction.  
- Fully connected layers for regression of steering angles.  
- Normalization and dropout layers to reduce overfitting.  

---

## 📂 Dataset
- Source: **[Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)**.  
- Contains images (center, left, right cameras) and a CSV file with driving parameters (steering angle, throttle, brake, speed).  

---

## 🔗 Google Colab Notebook
You can view and run the training notebook here:  
👉 [Colab Link](https://colab.research.google.com/drive/16_9MosrVpDQ8FaBAFlftF1hDynBbz5Co?usp=sharing)

---

## 📈 Results

### 🔹 Training Performance
![Training Loss](assets/training_loss.png)  
*(Training loss plot over epochs – replace with your plot image)*  

### 🔹 Driving Demo
![Driving Demo](assets/driving_demo.gif)  
*(Autonomous driving demo from simulator – replace with your GIF/video screenshot)*  

---

## 📑 References
- NVIDIA Paper: [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)  
- Udacity Self-Driving Car Simulator  

---

## 🚀 Future Work
- Improve generalization with data augmentation.  
- Experiment with different deep learning architectures (ResNet, EfficientNet).  
- Test real-world deployment on RC cars or Jetson Nano.  

---

## ⚡ Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/self-driving-car-simulator.git
