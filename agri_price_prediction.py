import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Crop Price Prediction",
    page_icon="🌾",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to check if model file exists
def model_exists(model_path):
    return os.path.exists(model_path)

# Function to train model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to save model
def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

# Function to load model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Add custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #3366FF;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.8rem;
    color: #0099CC;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.info-text {
    padding: 1rem;
    border-radius: 5px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Main app header
st.markdown("<h1 class='main-header'>🌾 Agricultural Crop Price Prediction System</h1>", unsafe_allow_html=True)

# Navigation menu
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Data Exploration", "Model Training", "Price Prediction", "Market Insights"]
)

# States dictionary with major crops
states_dict = {
    "Andhra Pradesh": ["Rice", "Cotton", "Chillies", "Turmeric", "Sugarcane"],
    "Assam": ["Rice", "Tea", "Jute", "Sugarcane", "Oilseeds"],
    "Bihar": ["Rice", "Wheat", "Maize", "Pulses", "Sugarcane"],
    "Chhattisgarh": ["Rice", "Maize", "Pulses", "Oilseeds", "Wheat"],
    "Gujarat": ["Cotton", "Groundnut", "Wheat", "Bajra", "Sugarcane"],
    "Haryana": ["Wheat", "Rice", "Sugarcane", "Cotton", "Oilseeds"],
    "Himachal Pradesh": ["Apple", "Wheat", "Maize", "Potato", "Ginger"],
    "Jharkhand": ["Rice", "Maize", "Pulses", "Wheat", "Oilseeds"],
    "Karnataka": ["Rice", "Ragi", "Jowar", "Coffee", "Sugarcane"],
    "Kerala": ["Coconut", "Rice", "Rubber", "Spices", "Banana"],
    "Madhya Pradesh": ["Wheat", "Soybean", "Pulses", "Rice", "Cotton"],
    "Maharashtra": ["Jowar", "Cotton", "Sugarcane", "Soybean", "Rice"],
    "Odisha": ["Rice", "Pulses", "Oilseeds", "Jute", "Sugarcane"],
    "Punjab": ["Wheat", "Rice", "Cotton", "Sugarcane", "Maize"],
    "Rajasthan": ["Wheat", "Barley", "Pulses", "Oilseeds", "Cotton"],
    "Tamil Nadu": ["Rice", "Sugarcane", "Coconut", "Cotton", "Groundnut"],
    "Telangana": ["Rice", "Cotton", "Maize", "Pulses", "Chillies"],
    "Uttar Pradesh": ["Wheat", "Sugarcane", "Rice", "Pulses", "Potato"],
    "Uttarakhand": ["Rice", "Wheat", "Pulses", "Oilseeds", "Sugarcane"],
    "West Bengal": ["Rice", "Jute", "Potato", "Tea", "Oilseeds"]
}

# Current market trends and factors (simulated data)
market_trends = {
    "Rice": {"trend": "Increasing", "factors": ["Drought in key growing regions", "Increased export demand", "Lower production estimates"]},
    "Wheat": {"trend": "Stable", "factors": ["Adequate monsoon in wheat belt", "Balanced supply-demand", "Government MSP support"]},
    "Cotton": {"trend": "Decreasing", "factors": ["Higher production estimates", "Reduced international demand", "Increased competition"]},
    "Sugarcane": {"trend": "Increasing", "factors": ["Higher ethanol demand", "Lower Brazilian production", "Government incentives"]},
    "Maize": {"trend": "Stable", "factors": ["Increased feed demand", "Average production estimates", "Import restrictions"]},
    "Pulses": {"trend": "Increasing", "factors": ["Lower buffer stocks", "Reduced imports", "Crop damage in key regions"]},
    "Oilseeds": {"trend": "Increasing", "factors": ["Edible oil price rise", "Import restrictions", "Lower international production"]},
    "Potato": {"trend": "Decreasing", "factors": ["Bumper harvest", "Storage issues", "Limited export opportunities"]},
    "Onion": {"trend": "Volatile", "factors": ["Seasonal supply fluctuations", "Transportation challenges", "Storage constraints"]},
    "Tomato": {"trend": "Increasing", "factors": ["Crop damage in key regions", "Higher demand", "Transport cost increases"]},
}

# Home page
if menu == "Home":
    st.markdown("""
    <div class='info-text'>
    <p>Welcome to the Agricultural Crop Price Prediction System! This application helps farmers, traders, and policymakers predict crop prices based on historical data and current market factors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>How to use this application:</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <ol>
        <li><strong>Data Exploration:</strong> Upload and analyze your agricultural price dataset</li>
        <li><strong>Model Training:</strong> Train a machine learning model on your data</li>
        <li><strong>Price Prediction:</strong> Get price predictions for various crops</li>
        <li><strong>Market Insights:</strong> View current market trends and factors</li>
        </ol>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://wallpaperaccess.com/full/3543885.jpg", width=400, caption="Crop Price Prediction Process")
    
    st.info("Start by uploading your dataset in the Data Exploration section.")

# Data Exploration
elif menu == "Data Exploration":
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your agricultural price dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['data'] = df
            st.success(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns!")
            
            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head())
            
            st.markdown("<h3>Data Summary</h3>", unsafe_allow_html=True)
            st.write(df.describe())
            
            st.markdown("<h3>Check for Missing Values</h3>", unsafe_allow_html=True)
            missing_values = df.isnull().sum()
            st.write(missing_values)
            
            if missing_values.sum() > 0:
                st.warning("Your dataset contains missing values. Consider handling them before model training.")
            
            # Simple visualization
            st.markdown("<h3>Data Visualization</h3>", unsafe_allow_html=True)
            
            if 'Price' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['Price'], kde=True, ax=ax)
                plt.title('Distribution of Crop Prices')
                plt.xlabel('Price')
                plt.ylabel('Frequency')
                st.pyplot(fig)
                
                # Show price trends if time-related column exists
                time_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time'])]
                
                if time_cols:
                    st.markdown("<h3>Price Trends Over Time</h3>", unsafe_allow_html=True)
                    time_col = time_cols[0]
                    
                    if 'Crop' in df.columns:
                        crops = df['Crop'].unique()
                        selected_crops = st.multiselect('Select crops to visualize', options=crops, default=crops[:3] if len(crops) > 3 else crops)
                        
                        if selected_crops:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            for crop in selected_crops:
                                crop_data = df[df['Crop'] == crop]
                                ax.plot(crop_data[time_col], crop_data['Price'], label=crop)
                            
                            plt.title('Price Trends by Crop')
                            plt.xlabel(time_col)
                            plt.ylabel('Price')
                            plt.legend()
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
    else:
        st.info("Please upload a dataset to begin exploration.")
        
        # Show sample dataset format
        st.markdown("<h3>Expected Dataset Format</h3>", unsafe_allow_html=True)
        sample_data = {
            'State': ['Maharashtra', 'Punjab', 'Karnataka', 'Tamil Nadu', 'Gujarat'],
            'District': ['Pune', 'Ludhiana', 'Mysore', 'Coimbatore', 'Ahmedabad'],
            'Market': ['Pune Mkt', 'Ludhiana Mkt', 'Mysore Mkt', 'Coimbatore Mkt', 'Ahmedabad Mkt'],
            'Crop': ['Wheat', 'Rice', 'Ragi', 'Sugarcane', 'Cotton'],
            'Variety': ['Common', 'Basmati', 'Local', 'Co-86032', 'Long Staple'],
            'Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
            'Price': [2200, 3500, 1800, 310, 6500],
            'Rainfall': [120, 95, 85, 110, 45],
            'Temperature': [28, 32, 30, 33, 36],
            'Soil_Moisture': [75, 65, 60, 70, 50]
        }
        st.dataframe(pd.DataFrame(sample_data))

# Model Training
elif menu == "Model Training":
    st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        st.markdown("<h3>Feature Selection</h3>", unsafe_allow_html=True)
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Select target variable
        if 'Price' in numeric_cols:
            target_col = 'Price'
        else:
            target_col = st.selectbox("Select target variable (price column):", numeric_cols)
        
        # Select features
        feature_cols = st.multiselect(
            "Select features for the model:",
            options=[col for col in df.columns if col != target_col],
            default=[col for col in numeric_cols if col != target_col]
        )
        
        if not feature_cols:
            st.warning("Please select at least one feature for training.")
        else:
            # Handle categorical features
            categorical_features = [col for col in feature_cols if col in categorical_cols]
            if categorical_features:
                st.write("Categorical features will be encoded:")
                st.write(categorical_features)
                
                # One-hot encode categorical features
                df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
                
                # Update feature columns to include encoded features
                feature_cols = [col for col in df_encoded.columns if col != target_col and col in df_encoded.columns]
            else:
                df_encoded = df.copy()
            
            # Prepare data for training
            X = df_encoded[feature_cols]
            y = df_encoded[target_col]
            
            # Train-test split
            test_size = st.slider("Test size (%):", 10, 40, 20) / 100
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a moment."):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        
                        model = train_model(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Evaluate model
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Save model and feature list
                        st.session_state['model'] = model
                        st.session_state['features'] = feature_cols
                        st.session_state['target'] = target_col
                        st.session_state['categorical_features'] = categorical_features
                        
                        # Display metrics
                        st.success("Model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mean Squared Error", f"{mse:.2f}")
                        col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
                        col3.metric("R² Score", f"{r2:.2f}")
                        
                        # Feature importance
                        st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
                        feature_importance = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
                        plt.title('Top 10 Feature Importance')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Actual vs Predicted
                        st.markdown("<h3>Actual vs Predicted Prices</h3>", unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        plt.xlabel('Actual Price')
                        plt.ylabel('Predicted Price')
                        plt.title('Actual vs Predicted Prices')
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error training model: {e}")
    else:
        st.warning("Please upload and explore a dataset first in the Data Exploration section.")

# Price Prediction
elif menu == "Price Prediction":
    st.markdown("<h2 class='sub-header'>Crop Price Prediction</h2>", unsafe_allow_html=True)
    
    if 'model' in st.session_state:
        # Get state and crop selection
        state = st.selectbox("Select State:", list(states_dict.keys()))
        crop = st.selectbox("Select Crop:", states_dict[state])
        
        # Get other features based on the model requirements
        feature_inputs = {}
        
        st.markdown("<h3>Enter Feature Values:</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Standard features that most models would use
            district = st.text_input("District:", "Sample District")
            market = st.text_input("Market:", "Sample Market")
            variety = st.text_input("Variety:", "Common")
            rainfall = st.slider("Rainfall (mm):", 0, 500, 100)
            temperature = st.slider("Temperature (°C):", 10, 50, 30)
        
        with col2:
            soil_moisture = st.slider("Soil Moisture (%):", 0, 100, 60)
            humidity = st.slider("Humidity (%):", 0, 100, 65)
            
            # Get current date and allow selection of prediction date
            current_date = datetime.now()
            prediction_date = st.date_input("Prediction Date:", current_date)
            
            # Season based on month
            months = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 
                     6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter"}
            season = months[prediction_date.month]
        
        # Prepare input features
        feature_inputs = {
            'State': state,
            'District': district,
            'Market': market,
            'Crop': crop,
            'Variety': variety,
            'Date': prediction_date.strftime('%Y-%m-%d'),
            'Rainfall': rainfall,
            'Temperature': temperature,
            'Soil_Moisture': soil_moisture,
            'Humidity': humidity,
            'Season': season
        }
        
        # Display current market trends
        if crop in market_trends:
            st.markdown("<h3>Current Market Trends:</h3>", unsafe_allow_html=True)
            
            trend_info = market_trends[crop]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if trend_info["trend"] == "Increasing":
                    st.markdown(f"<p>🔼 <strong>Trend:</strong> {trend_info['trend']}</p>", unsafe_allow_html=True)
                elif trend_info["trend"] == "Decreasing":
                    st.markdown(f"<p>🔽 <strong>Trend:</strong> {trend_info['trend']}</p>", unsafe_allow_html=True)
                elif trend_info["trend"] == "Volatile":
                    st.markdown(f"<p>↕️ <strong>Trend:</strong> {trend_info['trend']}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p>➡️ <strong>Trend:</strong> {trend_info['trend']}</p>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<p><strong>Influencing Factors:</strong></p>", unsafe_allow_html=True)
                for factor in trend_info["factors"]:
                    st.markdown(f"<p>• {factor}</p>", unsafe_allow_html=True)
        
        # Make prediction button
        if st.button("Predict Price"):
            try:
                # Create a dataframe from input features
                input_df = pd.DataFrame([feature_inputs])
                
                # Handle categorical features (similar to training process)
                categorical_features = st.session_state.get('categorical_features', [])
                if categorical_features:
                    input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
                
                # Ensure all required features are present
                required_features = st.session_state['features']
                for feature in required_features:
                    if feature not in input_df.columns:
                        input_df[feature] = 0  # Add missing dummy variables
                
                # Select only the features used in the model
                input_features = input_df[required_features]
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_features)[0]
                
                # Apply market trend adjustment
                if crop in market_trends:
                    trend = market_trends[crop]["trend"]
                    if trend == "Increasing":
                        adjustment = 1.05  # 5% increase
                    elif trend == "Decreasing":
                        adjustment = 0.95  # 5% decrease
                    elif trend == "Volatile":
                        adjustment = np.random.uniform(0.97, 1.03)  # Random adjustment
                    else:  # Stable
                        adjustment = 1.0
                    
                    adjusted_prediction = prediction * adjustment
                else:
                    adjusted_prediction = prediction
                
                # Display prediction
                st.success(f"Predicted Price: ₹{adjusted_prediction:.2f} per quintal")
                
                # Show price forecast for next few days
                st.markdown("<h3>Price Forecast (Next 7 Days):</h3>", unsafe_allow_html=True)
                
                # Generate forecasted prices with some randomness
                dates = [(prediction_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
                
                if crop in market_trends:
                    trend = market_trends[crop]["trend"]
                    if trend == "Increasing":
                        daily_changes = [np.random.uniform(0.0, 0.02) for _ in range(7)]  # 0-2% daily increase
                    elif trend == "Decreasing":
                        daily_changes = [np.random.uniform(-0.02, 0.0) for _ in range(7)]  # 0-2% daily decrease
                    elif trend == "Volatile":
                        daily_changes = [np.random.uniform(-0.03, 0.03) for _ in range(7)]  # -3% to +3% daily change
                    else:  # Stable
                        daily_changes = [np.random.uniform(-0.005, 0.005) for _ in range(7)]  # -0.5% to +0.5% daily change
                else:
                    daily_changes = [np.random.uniform(-0.01, 0.01) for _ in range(7)]
                
                # Calculate cumulative changes
                cumulative_changes = [1.0]
                for change in daily_changes:
                    cumulative_changes.append(cumulative_changes[-1] * (1 + change))
                cumulative_changes = cumulative_changes[1:]
                
                # Apply changes to prediction
                forecasted_prices = [adjusted_prediction * change for change in cumulative_changes]
                
                # Create dataframe for forecasted prices
                forecast_df = pd.DataFrame({
                    'Date': dates,
                    'Forecasted Price': forecasted_prices
                })
                
                # Display as table
                st.dataframe(forecast_df)
                
                # Display as chart
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(forecast_df['Date'], forecast_df['Forecasted Price'], marker='o')
                plt.axhline(y=adjusted_prediction, color='r', linestyle='--', label='Current Prediction')
                plt.xlabel('Date')
                plt.ylabel('Forecasted Price (₹)')
                plt.title(f'7-Day Price Forecast for {crop} in {state}')
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.warning("Please train a model first in the Model Training section.")

# Market Insights
elif menu == "Market Insights":
    st.markdown("<h2 class='sub-header'>Market Insights</h2>", unsafe_allow_html=True)
    
    # Display current market trends for all crops
    st.markdown("<h3>Current Market Trends by Crop</h3>", unsafe_allow_html=True)
    
    # Create a dataframe for better display
    trends_data = []
    for crop, info in market_trends.items():
        trends_data.append({
            "Crop": crop,
            "Trend": info["trend"],
            "Key Factors": ", ".join(info["factors"])
        })
    
    trends_df = pd.DataFrame(trends_data)
    
    # Add trend indicators
    def color_trends(val):
        if val == "Increasing":
            return 'background-color: rgba(76, 175, 80, 0.2)'
        elif val == "Decreasing":
            return 'background-color: rgba(244, 67, 54, 0.2)'
        elif val == "Volatile":
            return 'background-color: rgba(255, 152, 0, 0.2)'
        else:
            return 'background-color: rgba(33, 150, 243, 0.2)'
    
    # Display styled dataframe
    st.dataframe(trends_df.style.applymap(color_trends, subset=['Trend']))
    
    # Regional highlights
    st.markdown("<h3>Regional Market Highlights</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_state = st.selectbox("Select a state:", list(states_dict.keys()))
        
    with col2:
        selected_crop = st.selectbox("Select a crop:", states_dict[selected_state])
    
    # Generate some regional insights (simulated data)
    regional_insights = {
        "Production": f"{np.random.randint(80, 120)}% of last year",
        "Demand": f"{np.random.randint(90, 110)}% of last year",
        "Major Markets": ", ".join(np.random.choice(["Local", "Export", "Processing", "Urban Centers"], size=2, replace=False)),
        "Price Trend": np.random.choice(["Rising", "Falling", "Stable"]),
        "MSP Status": np.random.choice(["Above MSP", "Below MSP", "At MSP"]),
        "Storage Availability": f"{np.random.randint(60, 95)}%"
    }
    
    # Display insights
    st.markdown(f"<h4>Market Insights for {selected_crop} in {selected_state}</h4>", unsafe_allow_html=True)
    
    for key, value in regional_insights.items():
        st.markdown(f"<p><strong>{key}:</strong> {value}</p>", unsafe_allow_html=True)
    
    # Recommendations based on insights
    st.markdown("<h4>Recommendations:</h4>", unsafe_allow_html=True)
    
    if regional_insights["Price Trend"] == "Rising":
        st.markdown("""
        <ul>
          <li>Consider holding stock for higher returns</li>
          <li>Explore direct marketing to consumers</li>
          <li>Monitor daily price movements closely</li>
        </ul>
        """, unsafe_allow_html=True)
    elif regional_insights["Price Trend"] == "Falling":
        st.markdown("""
        <ul>
          <li>Consider early selling to minimize losses</li>
          <li>Explore value addition options</li>
          <li>Look for government procurement programs</li>
        </ul>
        """, unsafe_allow_html=True)
    else:  # Stable
        st.markdown("""
        <ul>
          <li>Maintain regular selling schedule</li>
          <li>Focus on quality improvement for better returns</li>
          <li>Explore forward contracts with buyers</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # News and updates section
    st.markdown("<h3>Latest Agricultural News and Updates</h3>", unsafe_allow_html=True)
    
    # Simulated news
    news_items = [
        {
            "title": "Government Announces New MSP for Kharif Crops",
            "date": "April 15, 2025",
            "summary": "The government has announced a 7-10% increase in the Minimum Support Price (MSP) for major kharif crops, including paddy, pulses, and oilseeds."
        },
        {
            "title": "IMD Predicts Normal Monsoon This Year",
            "date": "April 10, 2025",
            "summary": "The Indian Meteorological Department (IMD) has predicted a normal monsoon season this year, bringing relief to farmers across the country."
        },
        {
            "title": "Export Restrictions Lifted for Select Agricultural Commodities",
            "date": "April 5, 2025",
            "summary": "The government has lifted export restrictions on select agricultural commodities, opening up new market opportunities for farmers."
        },
        {
            "title": f"Production of {selected_crop} Expected to Increase in {selected_state}",
            "date": "April 1, 2025",
            "summary": f"According to state agricultural department estimates, {selected_crop} production in {selected_state} is expected to increase by 15% this year due to favorable weather conditions."
        }
    ]
    
    # Display news items
    for item in news_items:
        with st.expander(f"{item['title']} ({item['date']})"):
            st.write(item['summary'])

# Footer
st.markdown("""
<hr>
<div style="text-align: center; padding: 20px;">
    <p>Agricultural Crop Price Prediction System | Developed with ❤️ for Farmers</p>
    <p>© 2025 | For educational and informational purposes only</p>
</div>
""", unsafe_allow_html=True)

# Add custom CSS for responsive design
st.markdown("""
<style>
@media (max-width: 768px) {
    .main-header {
        font-size: 1.8rem;
    }
    .sub-header {
        font-size: 1.4rem;
    }
}

/* Improve button styling */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 5px;
    transition: background-color 0.3s;
}

.stButton>button:hover {
    background-color: #45a049;
}

/* Improve metric styling */
[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1E88E5;
}

/* Improve dataframe styling */
.dataframe {
    font-size: 0.9rem;
}

/* Improve expander styling */
.streamlit-expanderHeader {
    font-size: 1rem;
    font-weight: bold;
}

/* Custom tooltip for help text */
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted black;
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# Helper functions for data preprocessing that can be used across the app
def preprocess_data(df):
    """Preprocess the dataset for model training"""
    # Handle missing values
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert date columns to datetime
    date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time'])]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            # Extract useful date features
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_day'] = df[col].dt.day
            
            # Add season feature
            df[f'{col}_season'] = df[col].dt.month.map({
                1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
            })
        except:
            pass
    
    return df

def get_model_performance_explanation(r2):
    """Provide explanation of model performance based on R-squared value"""
    if r2 >= 0.8:
        return "Excellent model fit! The model explains most of the variation in crop prices."
    elif r2 >= 0.6:
        return "Good model fit. The model explains a significant portion of the price variation."
    elif r2 >= 0.4:
        return "Moderate model fit. Consider adding more features or trying different algorithms."
    else:
        return "Poor model fit. Consider collecting more data or exploring different modeling approaches."

# Add error handling for prediction when features don't match
def safe_predict(model, features, required_features):
    """Safely make predictions ensuring all required features are present"""
    missing_features = set(required_features) - set(features.columns)
    extra_features = set(features.columns) - set(required_features)
    
    if missing_features:
        # Add missing features with zeros
        for feature in missing_features:
            features[feature] = 0
    
    if extra_features:
        # Remove extra features
        features = features[required_features]
    
    # Ensure correct order of features
    features = features[required_features]
    
    return model.predict(features)

# Helper function to generate price forecasts
def generate_price_forecast(base_price, days, trend, volatility=0.01):
    """Generate forecasted prices based on trend and volatility"""
    if trend == "Increasing":
        daily_changes = [np.random.uniform(0.0, volatility * 2) for _ in range(days)]
    elif trend == "Decreasing":
        daily_changes = [np.random.uniform(-volatility * 2, 0.0) for _ in range(days)]
    elif trend == "Volatile":
        daily_changes = [np.random.uniform(-volatility * 3, volatility * 3) for _ in range(days)]
    else:  # Stable
        daily_changes = [np.random.uniform(-volatility / 2, volatility / 2) for _ in range(days)]
    
    # Calculate cumulative changes
    cumulative_changes = [1.0]
    for change in daily_changes:
        cumulative_changes.append(cumulative_changes[-1] * (1 + change))
    cumulative_changes = cumulative_changes[1:]
    
    # Apply changes to base price
    forecasted_prices = [base_price * change for change in cumulative_changes]
    
    return forecasted_prices

# Function to get seasonal price patterns
def get_seasonal_patterns(crop):
    """Return seasonal price patterns for different crops"""
    patterns = {
        "Rice": {
            "Spring": "Moderate prices due to balanced supply",
            "Summer": "Increasing prices as old stocks deplete",
            "Fall": "Decreasing prices due to new harvest",
            "Winter": "Stable to slightly increasing prices"
        },
        "Wheat": {
            "Spring": "Decreasing prices due to harvest season",
            "Summer": "Low prices during peak supply",
            "Fall": "Gradually increasing prices",
            "Winter": "Higher prices due to lower supply"
        },
        "Cotton": {
            "Spring": "Moderate to high prices",
            "Summer": "Decreasing prices as harvest approaches",
            "Fall": "Low prices during peak harvest",
            "Winter": "Gradually increasing prices"
        },
        "Sugarcane": {
            "Spring": "Moderate prices",
            "Summer": "Lower prices due to decreased demand",
            "Fall": "Increasing prices as crushing season begins",
            "Winter": "Peak prices during crushing season"
        },
        "Maize": {
            "Spring": "Higher prices before new crop",
            "Summer": "Decreasing prices as harvest begins",
            "Fall": "Low prices during peak harvest",
            "Winter": "Gradually increasing prices"
        }
    }
    
    # Default pattern for crops not in the dictionary
    default_pattern = {
        "Spring": "Prices vary based on supply and demand",
        "Summer": "Prices vary based on supply and demand",
        "Fall": "Prices vary based on supply and demand",
        "Winter": "Prices vary based on supply and demand"
    }
    
    return patterns.get(crop, default_pattern)

# Function to get policy impacts on crop prices
def get_policy_impacts():
    """Return information about policy impacts on crop prices"""
    return {
        "MSP Increase": "Higher floor prices for farmers, potentially increasing market prices",
        "Export Restrictions": "Lower demand leading to price decrease in domestic markets",
        "Import Duties": "Higher prices due to restricted competition from imports",
        "Direct Benefit Transfers": "Improved farmer income without direct impact on market prices",
        "Storage Subsidies": "Enables farmers to hold stock, potentially stabilizing prices",
        "Ethanol Blending": "Increased demand for sugarcane, potentially increasing prices"
    }

# Add this code if running the app directly
if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Agricultural Crop Price Prediction System started")
    
    # Check if model directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        logger.info("Created models directory")
    
    # Add demo mode option in sidebar
    if st.sidebar.checkbox("Enable Demo Mode", False):
        logger.info("Demo mode enabled")
        
        # Load sample data if not already loaded
        if 'data' not in st.session_state:
            # Create sample data
            np.random.seed(42)
            
            # Generate dates for past 3 years
            start_date = datetime(2022, 1, 1)
            dates = [start_date + timedelta(days=i) for i in range(365 * 3)]
            dates_str = [date.strftime('%Y-%m-%d') for date in dates]
            
            # Generate sample states, districts, markets, crops
            states = list(states_dict.keys())
            sample_districts = ["District_" + str(i) for i in range(1, 21)]
            sample_markets = ["Market_" + str(i) for i in range(1, 31)]
            
            # Flatten the crops list
            all_crops = []
            for crops in states_dict.values():
                all_crops.extend(crops)
            all_crops = list(set(all_crops))
            
            varieties = ["Common", "Premium", "Local", "Hybrid", "Traditional"]
            
            # Generate sample data
            num_samples = 10000
            sample_data = {
                'State': np.random.choice(states, num_samples),
                'District': np.random.choice(sample_districts, num_samples),
                'Market': np.random.choice(sample_markets, num_samples),
                'Crop': np.random.choice(all_crops, num_samples),
                'Variety': np.random.choice(varieties, num_samples),
                'Date': np.random.choice(dates_str, num_samples),
                'Price': np.random.uniform(1000, 10000, num_samples),
                'Rainfall': np.random.uniform(0, 500, num_samples),
                'Temperature': np.random.uniform(10, 45, num_samples),
                'Soil_Moisture': np.random.uniform(20, 90, num_samples),
                'Humidity': np.random.uniform(30, 95, num_samples)
            }
            
            # Create seasonal patterns
            date_objects = pd.to_datetime(sample_data['Date'])
            months = date_objects.dt.month
            
            # Add seasonal effect to prices
            for i, (crop, month) in enumerate(zip(sample_data['Crop'], months)):
                # Rice prices higher in summer
                if crop == 'Rice' and month in [5, 6, 7, 8]:
                    sample_data['Price'][i] *= np.random.uniform(1.1, 1.3)
                # Wheat prices higher in winter
                elif crop == 'Wheat' and month in [11, 12, 1, 2]:
                    sample_data['Price'][i] *= np.random.uniform(1.1, 1.25)
                # Cotton prices higher in winter
                elif crop == 'Cotton' and month in [12, 1, 2]:
                    sample_data['Price'][i] *= np.random.uniform(1.05, 1.2)
            
            # Create correlation between weather and prices
            for i in range(num_samples):
                # High rainfall generally reduces prices (oversupply)
                if sample_data['Rainfall'][i] > 300:
                    sample_data['Price'][i] *= np.random.uniform(0.8, 0.95)
                # Extreme temperatures can raise prices (crop stress)
                if sample_data['Temperature'][i] > 40:
                    sample_data['Price'][i] *= np.random.uniform(1.1, 1.25)
                # Low soil moisture raises prices (drought conditions)
                if sample_data['Soil_Moisture'][i] < 30:
                    sample_data['Price'][i] *= np.random.uniform(1.15, 1.3)
            
            # Create DataFrame
            sample_df = pd.DataFrame(sample_data)
            
            # Add to session state
            st.session_state['data'] = sample_df
            logger.info("Sample data created and loaded into session state")
            
            # Train a demo model
            features = ['Rainfall', 'Temperature', 'Soil_Moisture', 'Humidity']
            target = 'Price'
            
            # Prepare data
            X = sample_df[features]
            y = sample_df[target]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = train_model(X_train, y_train)
            
            # Save model to session state
            st.session_state['model'] = model
            st.session_state['features'] = features
            st.session_state['target'] = target
            st.session_state['categorical_features'] = []
            
            logger.info("Demo model trained and saved to session state")
    
    # Add About section in sidebar
    with st.sidebar.expander("About This App"):
        st.write("""
        This Agricultural Crop Price Prediction System helps farmers, traders, and policymakers predict
        crop prices based on historical data and current market factors.
        
        **Features:**
        - Data exploration and visualization
        - Machine learning model training
        - Price prediction with current market trend adjustments
        - Forecasting of future prices
        - Market insights and recommendations
        
        **Version:** 2.0.0
        """)
    
    # Add Resources in sidebar
    with st.sidebar.expander("Resources"):
        st.markdown("""
        - [Agricultural Market Information](https://agmarknet.gov.in/)
        - [Weather Forecasts](https://mausam.imd.gov.in/)
        - [Minimum Support Prices](https://agricoop.nic.in/)
        - [Crop Production Statistics](https://eands.dacnet.nic.in/)
        """)
    
    # Add Feedback section in sidebar
    with st.sidebar.expander("Provide Feedback"):
        st.text_area("Share your feedback or suggestions:", "")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")
