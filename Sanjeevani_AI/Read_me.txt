# 🌱 AI-Powered Crop Leaf Disease Detection & Smart Pesticide Recommendation System

### 📌 Overview  
This project is a **web-based intelligent crop health analysis system** where users can upload a crop leaf image to instantly identify its disease and receive complete field management guidance. Using **Machine Learning, Deep Learning (CNN), and live weather data**, the system suggests **accurate pesticide recommendations** and **optimal spray timings** to improve crop yield and reduce chemical misuse.

---

### 🎯 Key Features  
- 📤 **Leaf Image Upload** – Simple and user-friendly interface  
- 🧠 **CNN-Based Disease Prediction** – Detects plant leaf diseases with high accuracy  
- 🧪 **Pesticide Recommendations** – Suggests both organic and inorganic solutions  
- 🌦️ **Weather-Based Advisory** – Notifies farmers when to spray for maximum effectiveness  
- 📊 **Detailed Disease Info** – Symptoms, causes & prevention tips  
- ☁️ **Cloud / Local Model Support** – Can run model predictions via server  
- 📈 **Scalable Design** – Can add new crops & diseases easily  

---

### 🛠️ Technologies Used  

| Layer | Technology |
|------|------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python (Flask / Django) |
| Database | MySQL / SQLite |
| Machine Learning | TensorFlow / Keras – Convolutional Neural Network |
| Weather API | OpenWeather / WeatherStack API |
| Hosting | GitHub / Render / AWS / Local Deployment |

---

### 🧬 Deep Learning Model Architecture  
- Input: Crop leaf image  
- Preprocessing: Normalization, Augmentation  
- Algorithm: CNN (Convolutional Neural Network)  
- Output: Disease class + Confidence score  

Supported Crops (Customizable):
- 🌾 Rice
- 🍅 Tomato
- 🥔 Potato
- 🌶️ Chilli
(You can add more anytime)

---

### 🩺 Crop Advisory Output  
After disease prediction, user will get:

✔ Disease Name  
✔ Symptoms & Cause  
✔ Best Weather Condition to Spray  
✔ Organic & Chemical Pesticide Suggestions  
✔ Preventive Measures / Field Management Tips  

---

### 📂 Project Structure  
Sanjeevani_AI/
├── __pycache__/
├── models/
├── static/
│   ├── uploads/
│   │   ├── AI_Assisted_Farming_Success_V...
│   │   ├── AI_Robot_Cares_for_Sad_Tomato...
│   │   ├── logo.jpg
│   └── styles.css
├── templates/
│   ├── index.html
│   ├── login.html
│   └── signup.html
├── weather files/
│   ├── balanced_crop_pesticide_dataset.csv
│   ├── balanced_crop_pesticide_dataset1.csv
│   ├── Weather_daata.py
│   └── Weather_prediction.ipynb
├── app.py
└── ml_integration.py



---

### 🚀 How to Run Locally

```bash
# Clone the repository
git clone git clone https://github.com/VAMSISAKEPURAM/Sanjeevani_AI-.git
cd Sanjeevani_AI-


# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
Now open browser and go to:

👉 http://127.0.0.1:5000/
🌐 Future Enhancements

📱 Mobile App Support

📍 Location-based crop advisory

🗣️ Multilingual Voice Support for Farmers

🛰️ Integration with Satellite/Drone NDVI data

🧪 Fertilizer recommendation system

🤖 Auto model retraining with new datasets

👨‍💻 Author

Vamsi
MCA Graduate | AI & ML Enthusiast
Sri Venkateswara University, Tirupati

🤝 Contribution & Support

Pull Requests are welcome!
If you like this project, please ⭐ the repo!


🌟 Empowering Farmers with Artificial Intelligence for a Healthier Crop Future! 🌾
