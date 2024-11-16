# Use the official Python image.
FROM python:3.9

# Install required packages
RUN pip install fastapi uvicorn mlflow

# Copy the service script
COPY service.py /app/service.py

# Set the working directory
WORKDIR /app

# Expose the FastAPI default port
EXPOSE 8000

# Start the FastAPI service
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
