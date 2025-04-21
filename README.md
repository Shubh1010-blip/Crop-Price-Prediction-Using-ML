# 🌾 Agricultural Crop Price Prediction System

An interactive machine learning web app that predicts agricultural crop prices across Indian states. Designed to support farmers, traders, and policymakers with data-driven insights based on historical prices, environmental factors, and current market trends.

---

## 📌 Project Objectives

- Predict future crop prices using machine learning  
- Visualize historical trends and seasonal variations  
- Help stakeholders make informed decisions  
- Provide region-specific insights and forecasts  
- Recommend crop strategies based on trends

---

## 💻 Technology Stack

**Languages & Tools:**  
- Python 3.8+  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Pickle (for model persistence)

**Machine Learning:**  
- Random Forest Regressor  
- Feature Engineering  
- Evaluation Metrics: RMSE, R²  
- Cross-validation for robustness

---

## 🚀 Features

**Data Exploration**  
- Upload CSV/Excel crop price datasets  
- View summary statistics and data distributions  
- Visualize trends and detect missing values  
- Explore feature relationships  

**Model Training**  
- Select and engineer features  
- Train Random Forest model  
- Evaluate model performance  
- View feature importance  
- Save/load trained models  

**Price Prediction**  
- Select state and crop  
- Input rainfall, temperature, soil moisture, humidity  
- Get predicted price with confidence score  
- View 7-day trend forecast  
- Receive insights and recommendations  

**Market Insights**  
- Monitor real-time price trends  
- Explore region-wise and crop-wise data  
- Analyze seasonal patterns  
- Access agri-news and policy updates  

---

## 🌐 State & Crop Coverage

| Region  | States Covered  | Major Crops |
|---------|------------------|-------------|
| North   | Punjab, Haryana, UP, Rajasthan, Uttarakhand | Wheat, Rice, Sugarcane, Cotton, Barley |
| South   | Tamil Nadu, Karnataka, Kerala, AP, Telangana | Rice, Coffee, Spices, Coconut, Sugarcane |
| East    | Bengal, Bihar, Odisha, Jharkhand, Assam | Rice, Jute, Tea, Maize, Potatoes |
| West    | Maharashtra, Gujarat, MP, Chhattisgarh, HP | Cotton, Soybean, Jowar, Groundnut, Apples |

---

## 📊 Required Data Format

**Columns:**  
State, District, Market, Crop, Variety, Date, Price, Rainfall, Temperature, Soil_Moisture, Humidity

**Example Row:**  
Maharashtra, Pune, Pune Market, Wheat, Common, 2024-01-15, 2200, 120, 28, 75, 65

---

## 🌐 Live Application

🚀 [Click here to access the live Crop Price Prediction App](https://crop-price-prediction45.streamlit.app/)

## ⚙️ Getting Started

Clone the repository and run the application:

    git clone https://github.com/PiyushChauhan-web/Crop-Price-Prediction.git
    cd Crop-Price-Prediction

Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate         # On Windows: venv\\Scripts\\activate

Install the required libraries:

    pip install -r requirements.txt

Launch the app:

    streamlit run agri_price_prediction.py

---

## 🔍 How It Works

1. Preprocessing – Clean data, encode features  
2. Feature Engineering – Time, weather, and market-based features  
3. Model Training – Train Random Forest on historical data  
4. Prediction – Use current inputs to predict price  
5. Forecasting – Generate 7-day trend visualizations  

---

## 🔮 Future Enhancements

- Real-time Weather API integration  
- LSTM-based deep learning model for forecasting  
- Mobile application (Android/iOS)  
- Multi-language interface  
- SMS alerts for farmers  
- Government MSP & satellite imagery integration  

---

## 🤝 Contribution Guide

1. Fork the repository  
2. Create your branch: `git checkout -b feature/your-feature`  
3. Commit your changes: `git commit -m "Add feature"`  
4. Push to GitHub: `git push origin feature/your-feature`  
5. Open a Pull Request  

**Development Tips:**  
- Follow PEP 8 standards  
- Add docstrings and inline comments  
- Include test cases for new features  

---

## 📬 Contact & Support

For feedback or support, please open an issue on the GitHub repository.

---

<p align="center"><b>🚜 Made with ❤️ for Indian Agricultural Stakeholders</b></p>
"""
