# Face Mask Detector 😷

A real-time Face Mask Detection system using Python, OpenCV, and Deep Learning (MobileNetV2). This application detects faces via webcam and classifies them as "Mask" or "No Mask" in real-time.

## 🚀 Features
- **Real-time Detection**: Uses webcam feed to detect faces instantly.
- **Deep Learning Model**: powered by a pre-trained MobileNetV2 model tailored for mask detection.
- **Visual Feedback**:
  - 🟩 **Green Box**: Mask Detected
  - 🟥 **Red Box**: No Mask (Warning)
- **User-Friendly**: Simple interface with clear labels and confidence indicators.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Computer Vision**: OpenCV (`cv2`)
- **Deep Learning**: TensorFlow / Keras
- **Numerical Operations**: NumPy

## 📂 Project Structure
- `webcam_mask_detection.py`: Main script to run the real-time detection.
- `mask_detector_mobilenetv2.keras`: The trained deep learning model file.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar Cascade classifier for face detection.
- `requirements.txt`: List of Python dependencies.

## ⚙️ Installation

1. **Clone the Repository** (if applicable) or download the source code.

2. **Install Dependencies**
   Open your terminal or command prompt in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

1. **Run the Application**
   Execute the following command in your terminal:
   ```bash
   python webcam_mask_detection.py
   ```

2. **Webcam Feed**
   - The webcam will start automatically.
   - The system will draw a bounding box around detected faces.
   - **Green** indicates a mask is present.
   - **Red** indicates no mask is detected.

3. **Quit**
   - Press **`q`** on your keyboard or close the window to stop the application.

## ⚠️ Troubleshooting
- **Webcam not working?** Ensure no other application is using the camera.
- **Model not found?** Make sure `mask_detector_mobilenetv2.keras` is in the same directory as the script.
- **Haar Cascade error?** Verify `haarcascade_frontalface_default.xml` exists in the folder.

---
*Stay Safe! wear a mask.* 😷
