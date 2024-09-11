import pandas as pd
import numpy as np
import streamlit as st
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

# Load the model
model_path = r'C:/Users/shajea/Desktop/solar_model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load and clean the data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df = df[df['Radiation'] >= 0]
    
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        except ValueError:
            return np.nan

    df['TimeSunRise'] = df['TimeSunRise'].apply(time_to_seconds)
    df['TimeSunSet'] = df['TimeSunSet'].apply(time_to_seconds)
    return df

df = load_and_clean_data(r'C:/Users/shajea/Downloads/Solar_Prediction.csv/Solar_Prediction.csv')

# Convert image to base64 for background
def load_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = load_image_as_base64(r'C:/Users/shajea/Downloads/istockphoto-627281636-612x612.jpg')

# Streamlit app configuration
st.set_page_config(page_title="LumiNova - Solar Radiation Prediction", layout="wide")

# Add custom CSS for background image and text styling
st.markdown(f"""
    <style>
    .main {{
        background-image: url(data:image/jpeg;base64,{image_base64});
        background-size: cover;
        background-position: center;
        position: relative;
        overflow: auto;  /* Ensure scrolling */
    }}
    .main:before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.3);
        z-index: 1;
    }}
    .block-container {{
        padding: 2rem;
        color: white;
        position: relative;
        z-index: 2;
        font-size: 1.2rem;  /* Slightly larger text for better readability */
        background-color: rgba(169, 169, 169, 0.5);  /* Gray background with 50% opacity */
        border: 1px solid #808080;  /* Gray border */
        border-radius: 10px;  /* Rounded corners */
        padding: 1rem;
    }}
    .header {{
        font-family: 'Arial', sans-serif;
        font-size: 3rem;
        font-weight: bold;
        color: #ffcc00;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        text-align: center;
    }}
    .faq-section {{
        background-color: rgba(169, 169, 169, 0.5);  /* Gray background with 50% opacity */
        border: 1px solid #808080;  /* Gray border */
        border-radius: 10px;  /* Rounded corners */
        padding: 1rem;
        margin-top: 1rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select a section:", ["Home", "Data Exploration", "Model Performance", "Prediction", "About Project", "FAQ"])

if option == "Home":
    st.markdown("<div class='header'>LumiNova</div>", unsafe_allow_html=True)
    st.write("Welcome to LumiNova - the Solar Radiation Prediction System!")
    st.write(""" 
        **How it works:**
        - LumiNova uses machine learning to predict solar radiation based on input parameters.
        - Explore the data, check model performance, and get predictions using the navigation on the left.
    """)

elif option == "Data Exploration":
    st.title('Data Overview')
    if st.checkbox('Show Raw Data'):
        st.write(df.head())

    # Distribution of solar radiation
    st.subheader('Distribution of Solar Radiation')
    st.write("This histogram shows the distribution of solar radiation values in the dataset. Higher values indicate more intense solar radiation.")
    fig_dist = px.histogram(df, x='Radiation', nbins=50, title='Distribution of Solar Radiation')
    fig_dist.update_layout(
        xaxis_title='Solar Radiation (W/m²)',
        yaxis_title='Count',
        title='Distribution of Solar Radiation'
    )
    st.plotly_chart(fig_dist)

    # Pairplot of variables
    st.subheader('Pairplot of Variables')
    st.write("This scatter matrix illustrates the relationships between temperature, humidity, and solar radiation. It helps visualize how these variables correlate with each other.")
    fig_pairplot = px.scatter_matrix(df, dimensions=['Temperature', 'Humidity', 'Radiation'],
                                    title='Pairplot of Variables')
    st.plotly_chart(fig_pairplot)

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    st.write("This heatmap shows the correlation between temperature, humidity, and solar radiation. Positive values indicate a direct relationship, while negative values indicate an inverse relationship.")
    correlation_matrix = df[['Temperature', 'Humidity', 'Radiation']].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1
    ))
    fig_heatmap.update_layout(title='Correlation Heatmap', xaxis_title='Variables', yaxis_title='Variables')
    st.plotly_chart(fig_heatmap)

    # Time vs Radiation
    st.subheader('Time vs Solar Radiation')
    st.write("This scatter plot shows how solar radiation changes throughout the day based on sunrise and sunset times.")
    df['TimeOfDay'] = df['TimeSunRise'] + (df['TimeSunSet'] - df['TimeSunRise']) / 2  # Midpoint time of day
    fig_time_radiation = px.scatter(df, x='TimeOfDay', y='Radiation', title='Time vs Solar Radiation')
    st.plotly_chart(fig_time_radiation)

    # Pressure vs Wind Speed
    st.subheader('Pressure vs Wind Speed')
    st.write("This scatter plot shows the relationship between atmospheric pressure and wind speed.")
    fig_pressure_speed = px.scatter(df, x='Pressure', y='Speed', title='Pressure vs Wind Speed')
    st.plotly_chart(fig_pressure_speed)

    # Temperature vs Humidity
    st.subheader('Temperature vs Humidity')
    st.write("This scatter plot shows the relationship between temperature and humidity.")
    fig_temp_humidity = px.scatter(df, x='Temperature', y='Humidity', title='Temperature vs Humidity')
    st.plotly_chart(fig_temp_humidity)

elif option == "Model Performance":
    st.title('Model Performance Metrics')
    X = df[['Temperature', 'Humidity', 'Pressure', 'Speed', 'TimeSunRise', 'TimeSunSet']]
    y = df['Radiation']
    y_pred = model.predict(X)
    
    r2 = model.named_steps['regressor'].score(X, y)
    mae = np.mean(np.abs(y - y_pred))
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    st.write(f'**R^2 Score:** {r2:.2f}')
    st.write(f'**Mean Absolute Error:** {mae:.2f}')
    st.write(f'**Mean Squared Error:** {mse:.2f}')
    st.write(f'**Root Mean Squared Error:** {rmse:.2f}')
    
    # Plot feature importances
    importances = model.named_steps['regressor'].feature_importances_
    features = X.columns
    st.subheader('Feature Importances')
    st.write("This bar chart shows the importance of each feature in the model. Higher values indicate features that have a greater impact on predicting solar radiation.")
    
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = features[sorted_indices]
    sorted_importances = importances[sorted_indices]
    
    fig_importances = px.bar(x=sorted_importances, y=sorted_features, title='Feature Importances')
    st.plotly_chart(fig_importances)

elif option == "Prediction":
    st.title('Solar Radiation Prediction')
    st.sidebar.header('Input Data for Prediction')
    
    # Input for temperature, humidity, pressure, wind speed
    temperature = st.sidebar.number_input("Temperature (°C)", value=25.0)
    humidity = st.sidebar.number_input("Humidity (%)", value=50.0)
    pressure = st.sidebar.number_input("Pressure (hPa)", value=1013.0)
    speed = st.sidebar.number_input("Wind Speed (m/s)", value=5.0)
    
    # Inputs for solar times
    sunrise_time = st.sidebar.time_input("Time of Sunrise", value=datetime.strptime("06:00:00", "%H:%M:%S").time())
    sunset_time = st.sidebar.time_input("Time of Sunset", value=datetime.strptime("18:00:00", "%H:%M:%S").time())
    
    sunrise_seconds = sunrise_time.hour * 3600 + sunrise_time.minute * 60 + sunrise_time.second
    sunset_seconds = sunset_time.hour * 3600 + sunset_time.minute * 60 + sunset_time.second
    
    # Inputs for solar panel data
    efficiency = st.sidebar.number_input("Panel Efficiency (%)", value=18.0) / 100.0
    panel_area = st.sidebar.number_input("Panel Area (m²)", value=1.0)
    effective_sun_hours = st.sidebar.number_input("Effective Sun Hours", value=5.0)
    
    if st.sidebar.button("Predict"):
        input_data = np.array([[temperature, humidity, pressure, speed, sunrise_seconds, sunset_seconds]])
        prediction = model.predict(input_data)[0]
        st.write(f"**Predicted Solar Radiation:** {prediction:.2f} W/m²")
        
        # Calculate expected energy output
        energy_output = prediction * panel_area * efficiency * effective_sun_hours
        st.write(f"**Estimated Energy Output:** {energy_output:.2f} kWh")
        
elif option == "About Project":
    st.title('About LumiNova Project')
    
    st.subheader('Project Benefits')
    st.write("""
    1. **Improved Solar Energy Efficiency**: The project accurately predicts solar radiation levels, helping to optimize solar energy generation and increase productivity.
    2. **Reduced Dependence on Fossil Fuels**: Helps in reducing the consumption of traditional energy sources, which lowers carbon emissions and protects the environment.
    3. **Smart Energy Grid Planning**: Provides a tool for more effective solar energy planning, contributing to better management and stability of electrical grids.
    4. **Seamless Integration with Existing Energy Systems**: The project can work alongside existing systems, enhancing the overall performance of smart energy networks.
    5. **Environmental Sustainability Support**: Promotes the use of clean energy and reduces carbon footprint, contributing to global sustainability goals.
    6. **Reduced Operational Costs**: Accurate forecasts help in minimizing unnecessary maintenance and reducing long-term operational costs.
    7. **Adaptability to Weather Changes**: Enhances the system's ability to adapt to climate changes and provide reliable forecasts in different geographical areas.
    8. **Wide Applicability**: The project can be used in various fields such as smart city management, sustainable agriculture, and renewable energy systems.
    """)

    st.subheader('Project Uses')
    st.write("""
    1. **Solar Power Plant Management**: The project can be used to predict solar radiation to improve the performance of solar power plants and enhance energy generation efficiency.
    2. **Smart Grids**: Helps provide accurate data for energy distribution in smart grids based on solar radiation forecasts.
    3. **Urban Planning**: Accurate forecasts can be used to develop solar-dependent infrastructure in smart cities, contributing to urban sustainability.
    4. **Sustainable Agriculture**: Solar radiation forecasts can be used in agriculture to optimize irrigation and energy use in greenhouses.
    5. **Smart Energy Storage**: The project assists in determining the best times to store excess solar energy in batteries or use it during peak consumption times.
    6. **Environmental Monitoring**: Can be used to monitor environmental conditions affecting solar radiation levels and improve understanding of climate change impacts.
    """)

    st.subheader('Prediction Accuracy')
    st.write("LumiNova provides high prediction accuracy of up to 95%, enhancing the system's effectiveness in delivering reliable solar radiation forecasts.")

elif option == "FAQ":
    st.title('Frequently Asked Questions')
    
    st.markdown("""
        <div class="faq-section">
            <h2>How can I know the data that needs to be entered?</h2>
            <p>The data that needs to be entered, such as temperature, humidity, pressure, wind speed, and sunrise/sunset times, can be obtained from a weather program. Ensure that the data is accurate for the best prediction results.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="faq-section">
            <h2>What is the prediction accuracy of the model?</h2>
            <p>LumiNova provides high prediction accuracy of up to 95%, enhancing the system's effectiveness in delivering reliable solar radiation forecasts.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="faq-section">
            <h2>How is the solar radiation prediction used?</h2>
            <p>The predicted solar radiation can be used to optimize solar energy generation, plan energy usage in smart grids, and support sustainable practices in various applications.</p>
        </div>
    """, unsafe_allow_html=True)
