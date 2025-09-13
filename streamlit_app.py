# Enhanced Streamlit Salary Prediction App - Fixed with Immediate Updates
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Function to format Indian currency
def format_indian_currency(amount):
    """Format currency in Indian format (Lakhs, Crores)"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.1f}Cr"
    elif amount >= 100000:  # 1 Lakh  
        return f"₹{amount/100000:.1f}L"
    elif amount >= 1000:  # 1 Thousand
        return f"₹{amount/1000:.0f}K"
    else:
        return f"₹{amount:.0f}"

# Company tier mapping with detailed info
company_tier_mapping = {
    'google': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'microsoft': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'amazon': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'apple': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'meta': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'netflix': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'linkedin': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'adobe': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'salesforce': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},
    'uber': {'tier': 1, 'description': 'Tier 1 - Top Tech Giant'},

    'ibm': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},
    'accenture': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},
    'oracle': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},
    'sap': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},
    'deloitte': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},
    'cisco': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},
    'intel': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},
    'intuit': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},
    'vmware': {'tier': 2, 'description': 'Tier 2 - Major Enterprise'},

    'tcs': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'infosys': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'wipro': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'cts': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'hcl': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'capgemini': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'persistent': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'lti mindtree': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'tech mahindra': {'tier': 3, 'description': 'Tier 3 - IT Services'},
    'mphasis': {'tier': 3, 'description': 'Tier 3 - IT Services'},

    'virtusa': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},
    'hexaware': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},
    'zensar': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},
    'mindtek': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},
    'sonata software': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},
    'cybage': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},
    'birlasoft': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},
    'l&t infotech': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},

    'coforge': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'valuelabs': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'happiest minds': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'kpit': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'infogain': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'quest global': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'ust global': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'globallogic': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'cigniti': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
    'niit technologies': {'tier': 4, 'description': 'Tier 4 - Mid-size IT'},
    'itc infotech': {'tier': 5, 'description': 'Tier 5 - Emerging IT'},
}

# Location tier mapping with detailed descriptions
location_tier_map = {
    'Bangalore': {'tier': 1, 'description': 'Tier 1 (Metro - Tech Hub)'},
    'Hyderabad': {'tier': 1, 'description': 'Tier 1 (Metro - IT Hub)'}, 
    'Chennai': {'tier': 1, 'description': 'Tier 1 (Metro - IT Hub)'},
    'Mumbai': {'tier': 1, 'description': 'Tier 1 (Metro - Financial Hub)'},
    'Pune': {'tier': 2, 'description': 'Tier 2 (Major IT Hub)'},
    'Delhi': {'tier': 1, 'description': 'Tier 1 (Metro - Capital)'},
    'Gurgaon': {'tier': 2, 'description': 'Tier 2 (Major IT Hub)'},
    'Noida': {'tier': 2, 'description': 'Tier 2 (Major IT Hub)'}, 
    'Kolkata': {'tier': 2, 'description': 'Tier 2 (Major City)'},
    'Coimbatore': {'tier': 3, 'description': 'Tier 3 (Growing IT City)'},
    'Jaipur': {'tier': 3, 'description': 'Tier 3 (Emerging Hub)'},
    'Kochi': {'tier': 3, 'description': 'Tier 3 (IT Hub)'},
    'Indore': {'tier': 3, 'description': 'Tier 3 (Emerging Hub)'},
    'Ahmedabad': {'tier': 3, 'description': 'Tier 3 (Commercial Hub)'},
    'Bhubaneswar': {'tier': 3, 'description': 'Tier 3 (Government Hub)'}
}

# Load model & encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('salary_prediction_model.joblib')
        le_dict = joblib.load('label_encoders.joblib')
        return model, le_dict
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run 'python train_model.py' first to create the model files.")
        st.stop()

model, le_dict = load_model_and_encoders()

# Page configuration
st.set_page_config(
    page_title="💼 Enhanced Salary Predictor & Career Advisor", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for immediate updates
if 'current_company_tier' not in st.session_state:
    st.session_state.current_company_tier = None
if 'target_company_tier' not in st.session_state:
    st.session_state.target_company_tier = None
if 'location_tier' not in st.session_state:
    st.session_state.location_tier = None

# Custom CSS styling - FIXED DROPDOWN VISIBILITY
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 1.5rem;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-align: center;
        min-width: 200px;
    }
    .tier-1 { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
    .tier-2 { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; }
    .tier-3 { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; }
    .tier-4 { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; }
    .tier-5 { background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); color: white; }
    .form-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .chart-section {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }
    .insight-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .market-intelligence {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 2rem 0;
    }
    .market-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    /* FIX DROPDOWN VISIBILITY */
    .stSelectbox > div > div > div > div {
        background-color: white !important;
        color: #1e293b !important;
    }
    .stSelectbox > div > div > div {
        background-color: white !important;
        color: #1e293b !important;
    }
    .stSelectbox > div > div {
        background-color: white !important;
    }
    .stSelectbox label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    /* Make selected option visible */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #1e293b !important;
    }
    div[data-baseweb="select"] > div > div {
        color: #1e293b !important;
    }
    /* Company Tier Display Style */
    .company-tier {
        background: #e2e8f0;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: #334155;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">💼 Enhanced Salary Predictor & Career Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><strong>Built with ❤️ using Advanced Machine Learning & Location Intelligence</strong></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><em>Empowering IT professionals with data-driven career decisions</em></div>', unsafe_allow_html=True)

# TOP SECTION - Form Inputs with Immediate Tier Updates
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown("## 📊 Salary Prediction Input Form")

# Create placeholders for immediate tier updates
current_tier_placeholder = st.empty()
target_tier_placeholder = st.empty()
location_tier_placeholder = st.empty()

with st.form("prediction_form"):
    st.subheader("📝 Enter Your Professional Details:")

    # Create columns for better organization
    col1, col2 = st.columns(2)

    with col1:
        # Current Company selection
        st.markdown("**Current Company:**")
        current_company = st.selectbox("Select your current company", list(le_dict['current_company'].classes_), key="current_company")

        # Target Company selection
        st.markdown("**Target Company:**")
        target_company = st.selectbox("Select your target company", list(le_dict['target_company'].classes_), key="target_company")

        # Location selection
        st.markdown("**Location:**")
        location = st.selectbox("Select your location", list(le_dict['location'].classes_), key="location")

    with col2:
        years_of_experience = st.slider("Years of Experience", 0.0, 25.0, value=5.0, step=0.5)
        current_salary = st.number_input("Current Salary (Annual INR)", min_value=100000, value=1500000, step=50000)
        gender = st.selectbox("Gender", list(le_dict['gender'].classes_))
        current_role = st.selectbox("Current Role", list(le_dict['current_role'].classes_))
        education = st.selectbox("Education Level", list(le_dict['education'].classes_))

    # Additional details in full width
    st.markdown("**Additional Skills & Certifications:**")
    col3, col4 = st.columns(2)

    with col3:
        certifications = st.multiselect("Certifications", list(le_dict['certifications'].classes_))
    with col4:
        skills = st.multiselect("Skills", list(le_dict['skills'].classes_))

    # Submit button
    submitted = st.form_submit_button("🚀 Predict Expected Salary", use_container_width=True)

# IMMEDIATE TIER DISPLAY UPDATES (Outside the form for real-time updates)
with current_tier_placeholder.container():
    if current_company and current_company.lower() in company_tier_mapping:
        tier_info = company_tier_mapping[current_company.lower()]
        st.markdown(f'<div class="company-tier">Current Company Tier: {tier_info["tier"]} - {tier_info["description"]}</div>', unsafe_allow_html=True)
        st.session_state.current_company_tier = tier_info["tier"]
    elif current_company:
        st.markdown(f'<div class="company-tier">Current Company Tier: 5 - IT Company</div>', unsafe_allow_html=True)
        st.session_state.current_company_tier = 5

with target_tier_placeholder.container():
    if target_company and target_company.lower() in company_tier_mapping:
        tier_info = company_tier_mapping[target_company.lower()]
        st.markdown(f'<div class="company-tier">Target Company Tier: {tier_info["tier"]} - {tier_info["description"]}</div>', unsafe_allow_html=True)
        st.session_state.target_company_tier = tier_info["tier"]
    elif target_company:
        st.markdown(f'<div class="company-tier">Target Company Tier: 5 - IT Company</div>', unsafe_allow_html=True)
        st.session_state.target_company_tier = 5

with location_tier_placeholder.container():
    if location and location in location_tier_map:
        tier_info = location_tier_map[location]
        st.markdown(f'<div class="company-tier">Location Tier: {tier_info["tier"]} - {tier_info["description"]}</div>', unsafe_allow_html=True)
        st.session_state.location_tier = tier_info["tier"]
    elif location:
        st.markdown(f'<div class="company-tier">Location Tier: 3 - Other City</div>', unsafe_allow_html=True)
        st.session_state.location_tier = 3

st.markdown('</div>', unsafe_allow_html=True)

# Process prediction if form is submitted
if submitted:
    # Handle multi-selects (take 1st selected or 'None')  
    cert_input = certifications[0] if certifications else 'None'
    skill_input = skills[0] if skills else 'None'

    # Get company tiers
    current_tier = company_tier_mapping.get(current_company.lower(), {'tier': 5})['tier']
    target_tier = company_tier_mapping.get(target_company.lower(), {'tier': 5})['tier']

    # Build input DataFrame
    input_df = pd.DataFrame([{
        'current_company': current_company,
        'target_company': target_company,
        'years_of_experience': years_of_experience,
        'current_salary': current_salary,
        'gender': gender,
        'location': location,
        'current_role': current_role,
        'sector': 'IT',
        'education': education,
        'certifications': cert_input,
        'skills': skill_input,
        'current_company_tier': current_tier,
        'target_company_tier': target_tier
    }])

    # Encode categorical variables
    input_encoded = input_df.copy()
    for col in le_dict:
        if col in input_df.columns:
            try:
                input_encoded[col] = le_dict[col].transform(input_df[col])
            except ValueError:
                # Handle unseen categories
                input_encoded[col] = 0

    # Make prediction
    try:
        prediction = model.predict(input_encoded)[0]
        hike_percent = ((prediction - current_salary) / current_salary) * 100

        # Store prediction data for charts
        st.session_state.prediction_data = {
            'current_salary': current_salary,
            'predicted_salary': prediction,
            'current_company': current_company,
            'target_company': target_company,
            'location': location,
            'current_role': current_role,
            'hike_percent': hike_percent,
            'years_experience': years_of_experience,
            'current_tier': current_tier,
            'target_tier': target_tier
        }

    except Exception as e:
        st.error(f"Error making prediction: {e}")

# BOTTOM SECTION - Results, Charts and Analysis
if 'prediction_data' in st.session_state:
    data = st.session_state.prediction_data

    # Results Section
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown("## 🎯 Prediction Results")

    # Display main results with Indian currency formatting
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💼 Current Salary", 
            value=format_indian_currency(data['current_salary'])
        )

    with col2:
        st.metric(
            label="🚀 Predicted Salary", 
            value=format_indian_currency(data['predicted_salary']),
            delta=format_indian_currency(data['predicted_salary'] - data['current_salary'])
        )

    with col3:
        st.metric(
            label="📈 Expected Hike", 
            value=f"{data['hike_percent']:.1f}%"
        )

    with col4:
        tier_change = data['target_tier'] - data['current_tier']
        tier_direction = "⬆️" if tier_change < 0 else "⬇️" if tier_change > 0 else "➡️"
        st.metric(
            label="🏢 Tier Movement", 
            value=f"Tier {data['current_tier']} {tier_direction} Tier {data['target_tier']}"
        )

    # Display insights based on hike percentage
    if data['hike_percent'] > 50:
        st.balloons()
        st.success("🎉 **Excellent Opportunity!** This move could significantly boost your career trajectory!")
    elif data['hike_percent'] > 25:
        st.success("👍 **Great Opportunity!** This is well above market average growth.")
    elif data['hike_percent'] > 10:
        st.info("✅ **Good Opportunity** with decent growth potential.")
    else:
        st.warning("⚠️ **Consider Negotiating** or exploring additional opportunities for better growth.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Market Intelligence Section (Added from your second image)
    st.markdown("""
    <div class="market-intelligence">
        <h2>📚 Market Intelligence</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="market-card">
            <h4>🎯 Career Tips:</h4>
            <ul>
                <li>Tier 1 companies offer 50-80% higher packages</li>
                <li>Cloud & AI skills command premium salaries</li>
                <li>Certifications boost salary by 20-30%</li>
                <li>Leadership skills crucial for senior roles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="market-card">
            <h4>📈 Market Trends 2025:</h4>
            <ul>
                <li>DevOps Engineers: 40%+ hikes possible</li>
                <li>ML Engineers: 60-100% for experienced pros</li>
                <li>Full Stack: Steady 25-35% growth</li>
                <li>Cloud Architects: Premium 80%+ packages</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="market-card">
            <h4>💡 Negotiation Strategy:</h4>
            <ul>
                <li>Research market rates thoroughly</li>
                <li>Highlight unique skills & achievements</li>
                <li>Consider total compensation package</li>
                <li>Be prepared with multiple offers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Charts Section with Indian Currency Formatting (FIXED PLOTLY ERROR)
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown("## 📊 Salary Analysis Charts")

    # Chart 1: Current vs Predicted Salary Comparison with Indian Currency Format
    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure(data=[
            go.Bar(
                name='Current Salary',
                x=['Salary Comparison'],
                y=[data['current_salary']],
                marker_color='#ef4444',
                text=[format_indian_currency(data['current_salary'])],
                textposition='auto',
                width=0.4
            ),
            go.Bar(
                name='Predicted Salary',
                x=['Salary Comparison'],
                y=[data['predicted_salary']],
                marker_color='#10b981',
                text=[format_indian_currency(data['predicted_salary'])],
                textposition='auto',
                width=0.4
            )
        ])

        fig1.update_layout(
            title=f'💰 Current vs Predicted Salary<br><sub>{data["current_company"]} → {data["target_company"]}</sub>',
            xaxis_title='',
            yaxis_title='Annual Salary',
            barmode='group',
            height=400,
            showlegend=True,
            xaxis={'categoryorder': 'category ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # REMOVED THE PROBLEMATIC LINE: fig1.update_yaxis(tickformat='none')

        # Add hike percentage annotation
        fig1.add_annotation(
            x=0,
            y=max(data['current_salary'], data['predicted_salary']) * 1.1,
            text=f"<b>Expected Hike: {data['hike_percent']:.1f}%</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#667eea",
            bgcolor="rgba(240, 249, 255, 0.8)",
            bordercolor="#3b82f6",
            borderwidth=2,
            font=dict(size=12, color="#1e40af")
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Chart 2: Location-based salary analysis with Indian Currency Format
        locations = ['Bangalore', 'Hyderabad', 'Chennai', 'Mumbai', 'Pune', 'Delhi', 'Gurgaon', 'Noida', 'Kolkata']
        base_salary = data['predicted_salary']

        # Location multipliers based on market data
        location_multipliers = {
            'Bangalore': 1.0, 'Hyderabad': 0.95, 'Chennai': 0.92, 'Mumbai': 0.98,
            'Pune': 0.88, 'Delhi': 0.94, 'Gurgaon': 0.96, 'Noida': 0.90, 'Kolkata': 0.82
        }

        location_salaries = []
        colors = []
        for loc in locations:
            multiplier = location_multipliers.get(loc, 0.85)
            salary = base_salary * multiplier * (1 + np.random.uniform(-0.05, 0.05))
            location_salaries.append(salary)
            # Highlight selected location
            colors.append('#3b82f6' if loc == data['location'] else '#94a3b8')

        fig2 = go.Figure(data=[
            go.Bar(
                x=locations,
                y=location_salaries,
                marker_color=colors,
                text=[format_indian_currency(salary) for salary in location_salaries],
                textposition='outside'
            )
        ])

        fig2.update_layout(
            title=f'🌍 {data["current_role"]} Salary by Location',
            xaxis_title='Location',
            yaxis_title='Expected Salary',
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # REMOVED THE PROBLEMATIC LINE: fig2.update_yaxis(tickformat='none')

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Insights and Recommendations Section
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown("## 💡 Career Insights & Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Career Growth Analysis")
        if data['hike_percent'] > 30:
            st.markdown('<div class="insight-card">🚀 <strong>Excellent Growth:</strong> This move aligns with premium career progression in the IT industry.</div>', unsafe_allow_html=True)
        elif data['hike_percent'] > 15:
            st.markdown('<div class="insight-card">📈 <strong>Good Growth:</strong> Solid career advancement opportunity with competitive compensation.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-card">⚠️ <strong>Moderate Growth:</strong> Consider negotiating or exploring additional opportunities.</div>', unsafe_allow_html=True)

        # Tier movement analysis
        tier_change = data['target_tier'] - data['current_tier']
        if tier_change < 0:
            st.markdown('<div class="insight-card">⬆️ <strong>Tier Upgrade:</strong> Moving to a higher tier company - excellent for career growth!</div>', unsafe_allow_html=True)
        elif tier_change > 0:
            st.markdown('<div class="insight-card">⬇️ <strong>Tier Change:</strong> Consider the long-term impact on your career trajectory.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-card">➡️ <strong>Same Tier:</strong> Lateral movement within the same company tier.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 📍 Location Intelligence")
        if data['location'] in location_tier_map:
            loc_tier = location_tier_map[data['location']]['tier']
            if loc_tier == 1:
                st.markdown('<div class="insight-card">🏙️ <strong>Metro Advantage:</strong> You\'re in a Tier 1 city with maximum salary potential and opportunities.</div>', unsafe_allow_html=True)
            elif loc_tier == 2:
                st.markdown('<div class="insight-card">🌆 <strong>Major Hub:</strong> Good location with competitive salaries and growth opportunities.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-card">🏘️ <strong>Emerging Market:</strong> Consider relocating to Tier 1 cities for 15-25% salary boost.</div>', unsafe_allow_html=True)

        # Experience-based insights
        if data['years_experience'] < 3:
            st.markdown('<div class="insight-card">🌱 <strong>Early Career:</strong> Focus on skill development and gaining experience in trending technologies.</div>', unsafe_allow_html=True)
        elif data['years_experience'] < 8:
            st.markdown('<div class="insight-card">🚀 <strong>Growth Phase:</strong> Perfect time for strategic moves to accelerate career progression.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-card">👨‍💼 <strong>Senior Level:</strong> Focus on leadership roles and high-impact positions.</div>', unsafe_allow_html=True)

    with col3:
        st.markdown("### 🎓 Skill Enhancement")
        st.markdown('<div class="insight-card">☁️ <strong>Cloud Skills:</strong> AWS/Azure certifications can boost salary by 20-35%</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-card">🤖 <strong>AI/ML:</strong> Machine Learning expertise commands premium packages (50%+ hikes)</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-card">🔧 <strong>DevOps:</strong> DevOps engineers are in high demand with 40%+ growth potential</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-card">💼 <strong>Leadership:</strong> Management skills become crucial for 8+ years experience</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Show placeholder when no prediction is made
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown("## 📊 Salary Analysis")
    st.info("👆 **Please fill the form above and click 'Predict' to see your personalized salary analysis and recommendations!**")

    # Show sample visualization with Indian formatting
    st.markdown("### 📈 Sample Analysis Preview")
    sample_fig = go.Figure(data=[
        go.Bar(name='Current Salary', x=['Sample'], y=[1500000], marker_color='#ef4444', width=0.4,
               text=[format_indian_currency(1500000)], textposition='auto'),
        go.Bar(name='Target Salary', x=['Sample'], y=[2100000], marker_color='#10b981', width=0.4,
               text=[format_indian_currency(2100000)], textposition='auto')
    ])
    sample_fig.update_layout(
        title='💰 Salary Comparison Preview',
        height=300,
        showlegend=True,
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(sample_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit, Plotly, and Advanced ML Models | © 2025 Enhanced Salary Predictor*")
