# Experiment 10: Containerization of ML Application Using Docker

## Aim
To containerize the Iris classification machine learning application
using Docker and deploy it as a REST API.

## Technologies Used
- Python 3.11
- Docker
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTTP Server

## Implementation

1. Prepared the Iris feature dataset.
2. Trained a Random Forest classification model.
3. Saved the trained model and label encoder using Joblib.
4. Created a prediction server using Python HTTP Server.
5. Created requirements-serving.txt.
6. Created a multi-stage Dockerfile.
7. Created .dockerignore.
8. Built the Docker image.

## Docker Commands

Build:
docker build -t iris-classifier:v1 .

Run:
docker run -d --name iris-classifier-container -p 8000:8000 iris-classifier:v1

Check:
docker ps

View logs:
docker logs iris-classifier-container

Stop:
docker stop iris-classifier-container

Remove:
docker rm iris-classifier-container

## API Endpoints

GET /health
Checks application health.

POST /predict
Predicts Iris species using seven input features.

## Testing Results

- Health endpoint returned HTTP 200.
- Valid prediction request returned HTTP 200.
- Invalid input returned HTTP 400.
- Docker image was built successfully.
- Container logs confirmed successful API requests.

## Conclusion

Successfully containerized the Iris classification machine
learning application using a multi-stage Docker build.
The application runs inside a Docker container and provides
health-check and prediction endpoints.
