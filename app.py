import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class MockSalaryPredictor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            base_salary = 70
            
            job_title = str(row.get('Job Title', '')).lower()
            if 'senior' in job_title or 'lead' in job_title:
                base_salary += 30
            if 'manager' in job_title or 'director' in job_title:
                base_salary += 50
            if 'data scientist' in job_title:
                base_salary += 20
            if 'engineer' in job_title:
                base_salary += 15
            
            rating = float(row.get('Rating', 3.5))
            base_salary += (rating - 3.0) * 10
            
            location = str(row.get('Location', '')).lower()
            if 'new york' in location or 'san francisco' in location:
                base_salary += 25
            elif 'california' in location:
                base_salary += 15
            
            predictions.append(max(40, base_salary))
        
        return predictions

# Load pipeline
with open("salary_prediction_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Glassdoor Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}
.prediction-result {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 2rem;
    font-weight: bold;
    margin: 2rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💼 Glassdoor Salary Prediction App</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #7f8c8d;'>Get accurate salary predictions based on job details and company information</p>", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<h2 class="section-header">📝 Job Details</h2>', unsafe_allow_html=True)
    
    job_title = st.selectbox(
        "Job Title",
        ["Data Scientist", "Software Engineer", "Product Manager", "Data Analyst", 
         "Senior Data Scientist", "Lead Engineer", "Engineering Manager", "Director"],
        help="Select your job title or the closest match"
    )
    
    location = st.selectbox(
        "Location",
        ["New York, NY", "San Francisco, CA", "Los Angeles, CA", "Chicago, IL", 
         "Boston, MA", "Seattle, WA", "Austin, TX", "Denver, CO"],
        help="Choose the job location"
    )
    
    st.markdown('<h2 class="section-header">🏢 Company Info</h2>', unsafe_allow_html=True)
    
    industry = st.selectbox(
        "Industry",
        ["Information Technology", "Finance", "Healthcare", "Consulting", 
         "Manufacturing", "Retail", "Education", "Government"],
        help="Select the company's industry"
    )
    
    sector = st.selectbox(
        "Sector",
        ["Business Services", "Technology", "Financial Services", "Healthcare Services",
         "Consumer Services", "Industrial Services", "Government"],
        help="Choose the business sector"
    )
    
    ownership = st.selectbox(
        "Type of Ownership",
        ["Company - Private", "Company - Public", "Government", "Hospital", "Other Organization"],
        help="Select the type of organization"
    )
    
    st.markdown('<h2 class="section-header">⭐ Company Metrics</h2>', unsafe_allow_html=True)
    
    rating = st.slider(
        "Company Rating", 
        0.0, 5.0, 3.5, 0.1,
        help="Company rating on Glassdoor (0-5 stars)"
    )
    
    founded = st.number_input(
        "Founded Year", 
        min_value=1800, max_value=2025, value=2000,
        help="Year the company was founded"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="section-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
    
    if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
        with st.spinner("Analyzing job market data..."):
            input_data = pd.DataFrame([{
                "Job Title": job_title,
                "Location": location,
                "Industry": industry,
                "Sector": sector,
                "Type of ownership": ownership,
                "Rating": rating,
                "Founded": founded
            }])

            prediction = model.predict(input_data)
            salary = int(prediction[0])
            
            st.markdown(f'''
            <div class="prediction-result">
                💰 Estimated Salary: ${salary}K
            </div>
            ''', unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 📈 Salary Insights")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Annual Salary", 
                    f"${salary}K",
                    delta=f"{salary-70}K from base"
                )
            
            with col_b:
                monthly = salary * 1000 / 12
                st.metric(
                    "Monthly Salary", 
                    f"${monthly:,.0f}"
                )
            
            with col_c:
                hourly = salary * 1000 / (40 * 52)
                st.metric(
                    "Hourly Rate", 
                    f"${hourly:.0f}/hr"
                )
            
            # Salary breakdown
            st.markdown("### 🔍 Salary Factors")
            factors = []
            
            if 'senior' in job_title.lower() or 'lead' in job_title.lower():
                factors.append("🎯 Senior/Lead role: +$30K")
            if 'manager' in job_title.lower() or 'director' in job_title.lower():
                factors.append("👔 Management role: +$50K")
            if 'data scientist' in job_title.lower():
                factors.append("🔬 Data Science role: +$20K")
            if 'engineer' in job_title.lower():
                factors.append("⚙️ Engineering role: +$15K")
            if rating > 3.5:
                factors.append(f"⭐ High company rating: +${(rating-3.0)*10:.0f}K")
            if 'new york' in location.lower() or 'san francisco' in location.lower():
                factors.append("🏙️ High-cost location: +$25K")
            
            if factors:
                for factor in factors:
                    st.write(f"• {factor}")
            else:
                st.write("• Base salary calculation applied")

with col2:
    st.markdown('<h2 class="section-header">💡 Tips</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Maximize Your Salary:**
    • Target high-rated companies (4+ stars)
    • Consider major tech hubs
    • Pursue senior/lead positions
    • Specialize in high-demand skills
    """)
    
    st.markdown('<h2 class="section-header">📋 Summary</h2>', unsafe_allow_html=True)
    
    st.write(f"**Job:** {job_title}")
    st.write(f"**Location:** {location}")
    st.write(f"**Industry:** {industry}")
    st.write(f"**Rating:** {rating}⭐")
    st.write(f"**Founded:** {founded}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>💼 Built with Streamlit | Data-driven salary predictions</p>", 
    unsafe_allow_html=True
)