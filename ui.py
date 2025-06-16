import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Insurance Premium Predictor - India",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# City tier configuration
tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

def get_city_tier(city):
    if city in tier_1_cities:
        return "Tier 1"
    elif city in tier_2_cities:
        return "Tier 2"
    else:
        return "Tier 3"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86C1;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF6B35;
        padding-bottom: 5px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #E8F6F3;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1ABC9C;
        margin: 10px 0;
    }
    .tier-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .tier-1 { background-color: #E74C3C; color: white; }
    .tier-2 { background-color: #F39C12; color: white; }
    .tier-3 { background-color: #27AE60; color: white; }
</style>
""", unsafe_allow_html=True)

# Title with Indian flag colors theme
st.markdown('<h1 class="main-header">🏥 Insurance Premium Predictor - India</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7F8C8D; font-size: 1.2rem;">Predict your health insurance premium based on Indian market factors</p>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Choose a page", ["🔮 Premium Prediction", "📊 Data Analysis", "📈 Model Performance", "🏙️ City Analysis"])

# Sample data creation function for Indian context
@st.cache_data
def load_indian_insurance_data():
    """Create sample Indian insurance data"""
    np.random.seed(42)
    n_samples = 2000
    
    # Define and normalize age probabilities
    raw_probs = [0.05 if i < 25 else 0.15 if i < 35 else 0.2 if i < 45 else 0.25 if i < 55 else 0.15 if i < 65 else 0.2 for i in range(18, 66)]
    normalized_probs = [p / sum(raw_probs) for p in raw_probs]
    ages = np.random.choice(range(18, 66), n_samples, p=normalized_probs)
    
    # Indian cities with realistic distribution
    all_cities = tier_1_cities + tier_2_cities + ["Tier3_City"] * 20
    cities = np.random.choice(all_cities, n_samples)
    
    data = {
        'age': ages,
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'bmi': np.random.normal(24, 4, n_samples),  # Indian average BMI
        'children': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
        'smoker': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),  # Lower smoking rate in India
        'city': cities,
        'occupation': np.random.choice(['IT', 'Business', 'Government', 'Healthcare', 'Education', 'Other'], n_samples),
        'annual_income': np.random.lognormal(np.log(500000), 0.8, n_samples)  # Income in INR
    }
    
    df = pd.DataFrame(data)
    df['bmi'] = np.clip(df['bmi'], 16, 45)
    df['annual_income'] = np.clip(df['annual_income'], 200000, 5000000)
    
    # Add city tier
    df['city_tier'] = df['city'].apply(get_city_tier)
    
    # Calculate premium with Indian factors (in INR)
    base_premium = 15000  # Base premium in INR
    
    premium = (
        base_premium +
        (df['age'] - 25) * 800 +  # Age factor
        (df['bmi'] - 23) * 1500 +  # BMI factor (Indian normal BMI ~23)
        df['children'] * 8000 +  # Children factor
        (df['smoker'] == 'Yes') * 35000 +  # Smoking premium
        (df['gender'] == 'Male') * 5000 +  # Gender factor
        (df['city_tier'] == 'Tier 1') * 15000 +  # City tier factor
        (df['city_tier'] == 'Tier 2') * 8000 +
        (df['annual_income'] / 100000) * 2000 +  # Income factor
        np.random.normal(0, 5000, n_samples)  # Random variation
    )
    
    df['premium'] = np.maximum(premium, 8000)  # Minimum premium 8000 INR
    return df

# Prepare data for modeling
@st.cache_data
def prepare_indian_data():
    """Prepare Indian insurance data for modeling"""
    df = load_indian_insurance_data()
    
    # Create label encoders
    le_gender = LabelEncoder()
    le_smoker = LabelEncoder()
    le_city_tier = LabelEncoder()
    le_occupation = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['smoker_encoded'] = le_smoker.fit_transform(df['smoker'])
    df['city_tier_encoded'] = le_city_tier.fit_transform(df['city_tier'])
    df['occupation_encoded'] = le_occupation.fit_transform(df['occupation'])
    
    return df, (le_gender, le_smoker, le_city_tier, le_occupation)

# Train model
@st.cache_resource
def train_indian_model():
    """Train the Indian insurance prediction model"""
    df, encoders = prepare_indian_data()
    
    # Features for training
    features = ['age', 'gender_encoded', 'bmi', 'children', 'smoker_encoded', 
                'city_tier_encoded', 'occupation_encoded', 'annual_income']
    X = df[features]
    y = df['premium']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, (mae, mse, r2), (X_test, y_test, y_pred), encoders

# Initialize model and data
df, encoders = prepare_indian_data()
model, metrics, test_data, _ = train_indian_model()
le_gender, le_smoker, le_city_tier, le_occupation = encoders

# Page 1: Premium Prediction
if page == "🔮 Premium Prediction":
    st.markdown('<h2 class="sub-header">💰 Get Your Insurance Premium Quote</h2>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Personal Information")
        
        age = st.slider("Age", min_value=18, max_value=65, value=30, step=1)
        gender = st.selectbox("Gender", options=['Male', 'Female'])
        bmi = st.number_input("BMI (Body Mass Index)", min_value=16.0, max_value=45.0, value=23.0, step=0.1)
        
        # BMI calculator
        with st.expander("📏 Calculate BMI"):
            height = st.number_input("Height (cm)", min_value=140, max_value=200, value=170)
            weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70)
            if st.button("Calculate BMI"):
                calculated_bmi = weight / ((height/100) ** 2)
                st.success(f"Your BMI is: {calculated_bmi:.1f}")
                bmi = calculated_bmi
        
        st.subheader("👨‍👩‍👧‍👦 Family & Lifestyle")
        children = st.selectbox("Number of Dependents", options=[0, 1, 2, 3, 4])
        smoker = st.selectbox("Smoking Status", options=['No', 'Yes'])
        
    with col2:
        st.subheader("🏙️ Location & Work")
        
        # City selection with search
        selected_city = st.selectbox(
            "Select Your City", 
            options=["Select City"] + tier_1_cities + tier_2_cities + ["Other"],
            help="Choose your city to determine the tier-based premium"
        )
        
        if selected_city == "Other":
            custom_city = st.text_input("Enter your city name")
            if custom_city:
                selected_city = custom_city
        
        # Show city tier
        if selected_city != "Select City":
            city_tier = get_city_tier(selected_city)
            tier_color = "#E74C3C" if city_tier == "Tier 1" else "#F39C12" if city_tier == "Tier 2" else "#27AE60"
            st.markdown(f'<span class="tier-badge" style="background-color: {tier_color}; color: white;">{city_tier} City</span>', unsafe_allow_html=True)
        
        occupation = st.selectbox("Occupation", options=['IT', 'Business', 'Government', 'Healthcare', 'Education', 'Other'])
        
        annual_income = st.number_input(
            "Annual Income (₹)", 
            min_value=200000, 
            max_value=5000000, 
            value=500000, 
            step=50000,
            help="Enter your annual income in Indian Rupees"
        )
        
        # BMI category display
        st.subheader("📊 Health Assessment")
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "#3498DB"
        elif bmi < 25:
            bmi_category = "Normal weight"
            bmi_color = "#27AE60"
        elif bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "#F39C12"
        else:
            bmi_category = "Obese"
            bmi_color = "#E74C3C"
        
        st.markdown(f"""
        <div class="info-box" style="border-left-color: {bmi_color};">
            <h4 style="color: {bmi_color}; margin: 0;">{bmi_category}</h4>
            <p style="margin: 5px 0 0 0;">BMI: {bmi:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Assessment
    st.subheader("⚠️ Risk Assessment")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        risk_score = 0
        if smoker == 'Yes':
            risk_score += 3
        if bmi >= 30:
            risk_score += 2
        if age >= 50:
            risk_score += 2
        
        if risk_score >= 5:
            st.error("🔴 High Risk")
        elif risk_score >= 3:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")
    
    with col4:
        if smoker == 'Yes':
            st.write("🚬 Smoker")
        if bmi >= 30:
            st.write("⚖️ High BMI")
        if age >= 50:
            st.write("📅 Senior")
    
    with col5:
        city_tier = get_city_tier(selected_city) if selected_city != "Select City" else "Unknown"
        st.write(f"🏙️ {city_tier} City")
        st.write(f"💼 {occupation}")
    
    # Prediction
    st.markdown("---")
    
    if st.button("🔮 Calculate Premium", type="primary", use_container_width=True):
        if selected_city == "Select City":
            st.error("Please select your city to calculate premium")
        else:
            # Prepare input data
            city_tier = get_city_tier(selected_city)
            
            input_data = pd.DataFrame({
                'age': [age],
                'gender_encoded': [le_gender.transform([gender])[0]],
                'bmi': [bmi],
                'children': [children],
                'smoker_encoded': [le_smoker.transform([smoker])[0]],
                'city_tier_encoded': [le_city_tier.transform([city_tier])[0]],
                'occupation_encoded': [le_occupation.transform([occupation])[0]],
                'annual_income': [annual_income]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display main prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="margin-bottom: 20px;">💰 Your Annual Premium</h2>
                <h1 style="font-size: 3.5rem; margin: 0;">₹{prediction:,.0f}</h1>
                <p style="margin-top: 15px; opacity: 0.9;">Estimated annual health insurance premium</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional breakdowns
            col6, col7, col8 = st.columns(3)
            
            with col6:
                monthly_premium = prediction / 12
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="margin: 0;">Monthly Premium</h3>
                    <h2 style="margin: 10px 0 0 0;">₹{monthly_premium:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                quarterly_premium = prediction / 4
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="margin: 0;">Quarterly Premium</h3>
                    <h2 style="margin: 10px 0 0 0;">₹{quarterly_premium:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                income_percentage = (prediction / annual_income) * 100
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="margin: 0;">% of Income</h3>
                    <h2 style="margin: 10px 0 0 0;">{income_percentage:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Premium breakdown
            st.subheader("📋 Premium Breakdown")
            breakdown_data = {
                'Factor': ['Base Premium', 'Age Factor', 'BMI Factor', 'Smoking', 'City Tier', 'Gender', 'Dependents', 'Income Factor'],
                'Impact': ['₹15,000', f'₹{(age-25)*800:,}', f'₹{(bmi-23)*1500:,.0f}', 
                          f'₹{35000 if smoker=="Yes" else 0:,}', 
                          f'₹{15000 if city_tier=="Tier 1" else 8000 if city_tier=="Tier 2" else 0:,}',
                          f'₹{5000 if gender=="Male" else 0:,}', f'₹{children*8000:,}',
                          f'₹{(annual_income/100000)*2000:,.0f}']
            }
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            recommendations = []
            
            if smoker == 'Yes':
                recommendations.append("🚭 Quit smoking to significantly reduce your premium (save up to ₹35,000 annually)")
            
            if bmi >= 30:
                recommendations.append("🏃‍♂️ Maintain a healthy BMI to lower your premium")
            
            if city_tier == "Tier 1":
                recommendations.append("🏙️ Consider moving to Tier 2/3 cities for lower premiums")
            
            if not recommendations:
                recommendations.append("✅ You have an excellent risk profile! Consider higher coverage.")
            
            for rec in recommendations:
                st.info(rec)

# Page 2: Data Analysis
elif page == "📊 Data Analysis":
    st.markdown('<h2 class="sub-header">📊 Insurance Data Analysis</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Average Premium", f"₹{df['premium'].mean():,.0f}")
    with col3:
        st.metric("Age Range", f"{df['age'].min()}-{df['age'].max()}")
    with col4:
        st.metric("Cities Covered", df['city'].nunique())
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution Analysis", "🏙️ City Analysis", "👥 Demographics", "💰 Premium Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(df, x='age', title='Age Distribution', nbins=20)
            fig_age.update_layout(bargap=0.1)
            st.plotly_chart(fig_age, use_container_width=True)
            
            # BMI distribution
            fig_bmi = px.histogram(df, x='bmi', title='BMI Distribution', nbins=20)
            st.plotly_chart(fig_bmi, use_container_width=True)
        
        with col2:
            # Premium distribution
            fig_premium = px.histogram(df, x='premium', title='Premium Distribution', nbins=30)
            st.plotly_chart(fig_premium, use_container_width=True)
            
            # Children distribution
            fig_children = px.bar(df['children'].value_counts().reset_index(), 
                                 x='children', y='count', title='Number of Dependents')
            st.plotly_chart(fig_children, use_container_width=True)
    
    with tab2:
        # City tier analysis
        city_analysis = df.groupby('city_tier').agg({
            'premium': ['mean', 'median', 'count']
        }).round(0)
        city_analysis.columns = ['Average Premium', 'Median Premium', 'Count']
        st.subheader("Premium by City Tier")
        st.dataframe(city_analysis, use_container_width=True)
        
        # City tier premium comparison
        fig_city = px.box(df, x='city_tier', y='premium', title='Premium Distribution by City Tier')
        st.plotly_chart(fig_city, use_container_width=True)
        
        # Top cities by premium
        city_premium = df.groupby('city')['premium'].mean().sort_values(ascending=False).head(10)
        fig_top_cities = px.bar(x=city_premium.values, y=city_premium.index, 
                               orientation='h', title='Top 10 Cities by Average Premium')
        st.plotly_chart(fig_top_cities, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            gender_counts = df['gender'].value_counts()
            fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, 
                               title='Gender Distribution')
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Smoking status
            smoking_counts = df['smoker'].value_counts()
            fig_smoking = px.pie(values=smoking_counts.values, names=smoking_counts.index, 
                                title='Smoking Status')
            st.plotly_chart(fig_smoking, use_container_width=True)
        
        with col2:
            # Occupation distribution
            occupation_counts = df['occupation'].value_counts()
            fig_occupation = px.bar(x=occupation_counts.values, y=occupation_counts.index,
                                   orientation='h', title='Occupation Distribution')
            st.plotly_chart(fig_occupation, use_container_width=True)
            
            # Premium by gender and smoking
            fig_gender_smoking = px.box(df, x='gender', y='premium', color='smoker',
                                       title='Premium by Gender and Smoking Status')
            st.plotly_chart(fig_gender_smoking, use_container_width=True)
    
    with tab4:
        # Premium insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Age vs Premium
            fig_age_premium = px.scatter(df, x='age', y='premium', color='smoker',
                                        title='Age vs Premium (by Smoking Status)')
            st.plotly_chart(fig_age_premium, use_container_width=True)
            
            # Income vs Premium
            fig_income_premium = px.scatter(df, x='annual_income', y='premium', 
                                           color='city_tier', title='Income vs Premium (by City Tier)')
            st.plotly_chart(fig_income_premium, use_container_width=True)
        
        with col2:
            # BMI vs Premium
            fig_bmi_premium = px.scatter(df, x='bmi', y='premium', color='gender',
                                        title='BMI vs Premium (by Gender)')
            st.plotly_chart(fig_bmi_premium, use_container_width=True)
            
            # Premium by number of children
            fig_children_premium = px.box(df, x='children', y='premium',
                                         title='Premium by Number of Dependents')
            st.plotly_chart(fig_children_premium, use_container_width=True)

# Page 3: Model Performance
elif page == "📈 Model Performance":
    st.markdown('<h2 class="sub-header">📈 Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    mae, mse, r2 = metrics
    X_test, y_test, y_pred = test_data
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="margin: 0;">R² Score</h3>
            <h2 style="margin: 10px 0 0 0;">{r2:.3f}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8;">Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="margin: 0;">MAE</h3>
            <h2 style="margin: 10px 0 0 0;">₹{mae:,.0f}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8;">Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="margin: 0;">RMSE</h3>
            <h2 style="margin: 10px 0 0 0;">₹{np.sqrt(mse):,.0f}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8;">Root Mean Square Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction vs Actual plot
    fig_pred = px.scatter(x=y_test, y=y_pred, title='Predicted vs Actual Premium',
                         labels={'x': 'Actual Premium (₹)', 'y': 'Predicted Premium (₹)'})
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                       line=dict(color="red", width=2, dash="dash"))
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Residuals plot
    residuals = y_test - y_pred
    fig_residuals = px.scatter(x=y_pred, y=residuals, title='Residuals Plot',
                              labels={'x': 'Predicted Premium (₹)', 'y': 'Residuals (₹)'})
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        features = ['Age', 'Gender', 'BMI', 'Dependents', 'Smoker', 'City Tier', 'Occupation', 'Income']
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                               title='Feature Importance')
        st.plotly_chart(fig_importance, use_container_width=True)


# Page 4: City Analysis (continued)
elif page == "🏙️ City Analysis":
    st.markdown('<h2 class="sub-header">🏙️ City-wise Premium Analysis</h2>', unsafe_allow_html=True)
    
    # City tier comparison
    tier_comparison = df.groupby('city_tier').agg({
        'premium': ['mean', 'median', 'std', 'min', 'max'],
        'age': 'mean',
        'bmi': 'mean',
        'annual_income': 'mean'
    }).round(2)
    
    st.subheader("📊 City Tier Comparison")
    # Format the table for better readability
    tier_comparison.columns = [
        'Avg Premium (₹)', 'Median Premium (₹)', 'Premium Std Dev (₹)', 'Min Premium (₹)', 'Max Premium (₹)',
        'Avg Age', 'Avg BMI', 'Avg Income (₹)'
    ]
    st.dataframe(tier_comparison.style.format({
        'Avg Premium (₹)': '₹{:,.0f}',
        'Median Premium (₹)': '₹{:,.0f}',
        'Premium Std Dev (₹)': '₹{:,.0f}',
        'Min Premium (₹)': '₹{:,.0f}',
        'Max Premium (₹)': '₹{:,.0f}',
        'Avg Income (₹)': '₹{:,.0f}'
    }), use_container_width=True)
    
    # Visualizations for city tier analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for average premium by city tier
        fig_tier_premium = px.bar(
            tier_comparison.reset_index(),
            x='city_tier',
            y='Avg Premium (₹)',
            title='Average Premium by City Tier',
            color='city_tier',
            color_discrete_map={'Tier 1': '#E74C3C', 'Tier 2': '#F39C12', 'Tier 3': '#27AE60'}
        )
        st.plotly_chart(fig_tier_premium, use_container_width=True)
    
    with col2:
        # Box plot for premium distribution by city tier
        fig_tier_box = px.box(
            df,
            x='city_tier',
            y='premium',
            title='Premium Distribution by City Tier',
            color='city_tier',
            color_discrete_map={'Tier 1': '#E74C3C', 'Tier 2': '#F39C12', 'Tier 3': '#27AE60'}
        )
        st.plotly_chart(fig_tier_box, use_container_width=True)
    
    # City-specific analysis
    st.subheader("🔍 Analyze Specific Cities")
    selected_city_analysis = st.selectbox(
        "Select a City for Detailed Analysis",
        options=["Select City"] + sorted(df['city'].unique()),
        key="city_analysis_select"
    )
    
    if selected_city_analysis != "Select City":
        city_data = df[df['city'] == selected_city_analysis]
        city_tier = get_city_tier(selected_city_analysis)
        
        # Display city tier badge
        tier_color = "#E74C3C" if city_tier == "Tier 1" else "#F39C12" if city_tier == "Tier 2" else "#27AE60"
        st.markdown(
            f'<span class="tier-badge" style="background-color: {tier_color}; color: white;">{city_tier} City</span>',
            unsafe_allow_html=True
        )
        
        # City statistics
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Average Premium", f"₹{city_data['premium'].mean():,.0f}")
        with col4:
            st.metric("Median Premium", f"₹{city_data['premium'].median():,.0f}")
        with col5:
            st.metric("Number of Records", len(city_data))
        
        # Premium distribution for selected city
        fig_city_premium = px.histogram(
            city_data,
            x='premium',
            title=f'Premium Distribution in {selected_city_analysis}',
            nbins=20
        )
        st.plotly_chart(fig_city_premium, use_container_width=True)
        
        # Demographic breakdown
        st.subheader(f"📊 Demographics in {selected_city_analysis}")
        col6, col7 = st.columns(2)
        
        with col6:
            # Age distribution
            fig_city_age = px.histogram(
                city_data,
                x='age',
                title='Age Distribution',
                nbins=15
            )
            st.plotly_chart(fig_city_age, use_container_width=True)
        
        with col7:
            # Gender distribution
            gender_counts = city_data['gender'].value_counts()
            fig_city_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Gender Distribution'
            )
            st.plotly_chart(fig_city_gender, use_container_width=True)
    
    # Simulated map visualization (since Streamlit doesn't support maps natively)
    st.subheader("🗺️ City Tier Distribution")
    tier_counts = df['city_tier'].value_counts().reset_index()
    tier_counts.columns = ['City Tier', 'Count']
    
    fig_tier_pie = px.pie(
        tier_counts,
        values='Count',
        names='City Tier',
        title='Distribution of Records by City Tier',
        color='City Tier',
        color_discrete_map={'Tier 1': '#E74C3C', 'Tier 2': '#F39C12', 'Tier 3': '#27AE60'}
    )
    st.plotly_chart(fig_tier_pie, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #7F8C8D;">Developed by Insurance Analytics Team | Powered by Streamlit & Plotly</p>',
    unsafe_allow_html=True
)