# Income Prediction API.

This project provides an API to predict whether a person's annual income is greater than $50K based on demographic and financial features. It is built with Flask and uses a trained XGBoost model for classification.

---

## Description of the Problem.

Predicting a person's income level based on demographic and financial data is a common problem in data science and machine learning. Such predictions can be useful for.
- Targeted marketing campaigns.
- Risk assessment in financial sectors.
- Customized product offerings.

This project uses the Adult Income Dataset from the UCI Machine Learning Repository to train the model. The dataset includes attributes such as age, education, occupation, and more.

## How to Run the Project.

### 1. Prerequisites.

Ensure you have the following installed.
- Python 3.8+.
- Pip (Python package installer).
- Flask.
- Required libraries listed in `requirements.txt`.

### 2. Clone the Repository.

Clone the project repository from GitHub.

```bash
git clone https://github.com/ibacaraujo/midterm_project.git
cd midterm_project
```

### 3. Install Dependencies.

Install the necessary Python packages.

```bash
pip install -r requirements.txt
```

### 4. Start the Flask Server.

Run the Flask application.

```bash
python predict.py
```

### 5. Test the API.

You can test the API using the `predict_test.py` script.

```bash
python predict_test.py
```

### 6. Example Response.

The API will return a JSON response like this.

```json
{
    "income_probability": 0.72,
    "income": true
}
```

## Testing the Application with Containerization.

This section explains how to build, run, and test the application using Docker.

---

### 1. Build the Docker Image.

Navigate to the `midterm_project` folder and build the Docker image using the `Dockerfile`.

```bash
cd midterm_project
docker build -t midterm_project .
```

### 2. Run the Docker Container.

Start a container from the built image and map it to your local machine’s port:

```bash
docker run -d -p 8885:8885 midterm_project
```


### 3. Test the API

#### Test with the `predict_test.py` Script

Run the `predict_test.py` script to test the containerized API:

```bash
python predict_test.py
```

### 4. Test the Deployed Web Service.

You can test the deployed service using the following URL.

[https://midterm-project-xw26.onrender.com/predict](https://midterm-project-xw26.onrender.com/predict).

## Conclusion.

This project demonstrates a simple and effective way to deploy a machine learning model as a web service using Flask and Docker. The API provides predictions on whether a person's annual income exceeds $50K based on demographic and financial data, making it a valuable tool for various applications like targeted marketing and risk assessment.

Feel free to explore the deployed service and use the provided Docker setup to test the application locally or on other platforms.

---

For questions or suggestions, let's discuss. Obrigado.
