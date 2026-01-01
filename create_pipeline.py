import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class MockSalaryPredictor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        # Simple mock prediction based on job title and rating
        predictions = []
        for _, row in X.iterrows():
            base_salary = 70  # Base salary in thousands
            
            # Adjust based on job title
            job_title = str(row.get('Job Title', '')).lower()
            if 'senior' in job_title or 'lead' in job_title:
                base_salary += 30
            if 'manager' in job_title or 'director' in job_title:
                base_salary += 50
            if 'data scientist' in job_title:
                base_salary += 20
            if 'engineer' in job_title:
                base_salary += 15
            
            # Adjust based on rating
            rating = float(row.get('Rating', 3.5))
            base_salary += (rating - 3.0) * 10
            
            # Adjust based on location
            location = str(row.get('Location', '')).lower()
            if 'new york' in location or 'san francisco' in location:
                base_salary += 25
            elif 'california' in location:
                base_salary += 15
            
            predictions.append(max(40, base_salary))  # Minimum 40k
        
        return predictions

# Create and save the pipeline
pipeline = Pipeline([
    ('predictor', MockSalaryPredictor())
])

with open("salary_prediction_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Pipeline created successfully!")