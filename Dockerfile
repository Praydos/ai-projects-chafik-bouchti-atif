# Dockerfile
# Use a lightweight Python image as the base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy only the requirements file first to optimize layer caching
COPY requirements.txt .

# Install dependencies (Flask, Gunicorn, TF, Transformers)
# Installing numpy first can sometimes speed up the process
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# This includes app.py, templates/, and the saved_models/ directory
# Ensure your saved_models/ directory is in the project root!
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application using Gunicorn for production readiness
# 'app:app' means running the 'app' object inside 'app.py' module
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]