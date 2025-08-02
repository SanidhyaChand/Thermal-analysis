# Dockerfile

# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system libraries needed by OpenCV. This is the key step!
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose a port for the app to listen on
EXPOSE 10000

# The command to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]