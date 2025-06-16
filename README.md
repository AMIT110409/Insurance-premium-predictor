🏥 Insurance Premium Predictor
A FastAPI application that predicts health insurance premiums using a pre-trained ML model. Designed to be containerized with Docker and deployable to cloud environments.

<img width="958" alt="image" src="https://github.com/user-attachments/assets/f1f3b6af-b015-4283-85fe-47fecf5e32a3" />


📌 Table of Contents
Project Overview

Features

Architecture

Prerequisites

Local Setup & Usage

Docker Setup

API Endpoints

Testing

Deployment Recommendations

Contributing

License

🧠 Project Overview
A FastAPI-based microservice that exposes an endpoint for predicting insurance premium categories (e.g., low/medium/high), based on user features such as age, BMI, smoker status, city, income, and occupation.

The model is built, trained, and serialized with pickle, and integrated into the API for quick inference in real-world workflows.

✨ Features
FastAPI for high-performance model serving

Pydantic for input validation and computed feature calculations

Pandas for structured preprocessing

Pre-trained ML model (LightGBM / XGBoost, as used in your pipeline)

Dockerized with support for environment isolation

Ready for cloud deployment (e.g., AWS ECS, Azure Web App, GCP Run, or Kubernetes)

🏗️ Architecture
java
Copy
Edit
User Client
    │
    ▼
FastAPI App (app.py)
    ├─ Validates request via Pydantic
    ├─ Computes derived features (BMI, city tier, etc.)
    ├─ Loads pre-trained model (`model.pkl`)
    └─ Returns JSON { "predicted_category": "medium" }
✅ Prerequisites
Python ≥ 3.10 (ideally 3.11 for production stability)

Docker (for containerization)

git

🛠️ Local Setup & Usage
bash
Copy
Edit
git clone https://github.com/AMIT110409/Insurance-premium-predictor.git
cd Insurance-premium-predictor

python3 -m venv venv
source venv/bin/activate                # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

uvicorn app:app --reload               # Starts API at http://127.0.0.1:8000
🐳 Docker Setup
Build the image:

bash
Copy
Edit
docker build -t insurance-premium-predictor:latest .
Run the container:

bash
Copy
Edit
docker run -d --name ins-prem -p 8000:8000 insurance-premium-predictor:latest
Visit http://localhost:8000/docs for the auto-generated interactive API documentation.

🔌 API Endpoints
POST /predict
Accepts JSON payload:

json
Copy
Edit
{
  "age": 30,
  "weight": 70.5,
  "height": 1.7,
  "income_lpa": 12.0,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
Returns:

json
Copy
Edit
{
  "predicted_category": "medium"
}
🧪 Testing
Unit tests for validation logic (with pytest)

End-to-end tests using FastAPI’s TestClient

Load testing suggestions (e.g., locust, k6, wrk) to simulate usage and performance

🌐 Deployment Recommendations
Use a slim Python base image in Docker (e.g., python:3.11-slim)

Add Gunicorn / Uvicorn [workers], auto-reload off

Include health-check endpoints (/healthz) for Kubernetes

Configure environment variables for production

Store the model in S3, Azure Blob, or GCP Storage and load dynamically

🤝 Contributing
Fork the repo

Create a feature branch: git checkout -b feature/awesome

Commit changes: git commit -m "Add awesome feature"

Push branch: git push origin feature/awesome

Open a pull request

Please follow best practices for coding style, type annotations, and documentation.

📄 License
This project is licensed under MIT License. See the LICENSE file for details.

📧 Contact
Amit Rathore – Data Engineer & ML Enthusiast
Email: amitrathore110409@gmail.com 
