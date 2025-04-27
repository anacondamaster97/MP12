# Use an official Python runtime as a parent image
# NOTE: The template specified python:3.6, which is quite old.
# The requirements.txt was tested against Amazon Linux 2023.
# Consider using a newer version like python:3.9-slim or python:3.10-slim
# if you encounter dependency issues with 3.6. You might need to adjust
# requirements.txt based on the chosen Python version and base OS.
FROM python:3.6

# Creating Application Source Code Directory
# Use /app as a standard convention
RUN mkdir -p /app

# Setting Home Directory for containers
WORKDIR /app

# Copy requirements first to leverage Docker cache for dependencies
COPY requirements.txt .

# Install required python packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
# Ensure pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source python files from the current directory to /app in the container
# This includes classify.py, data_preload.py, train.py, models.py, utils.py
COPY *.py ./

# Create directories for models and data within the WORKDIR (/app)
# These directories will store the downloaded datasets and trained model files.
RUN mkdir -p ./data
RUN mkdir -p ./models

# Preload the data by running the data_preload.py script
# This downloads MNIST and KMNIST datasets into the ./data directory
RUN python data_preload.py

# Pretrain the models by running train.py for each combination.
# The trained models should be saved by train.py (likely into the ./models directory).
# Check train.py to confirm where it saves models.
# Training FeedForward model on MNIST
RUN python train.py --dataset mnist --type ff
# Training CNN model on MNIST
RUN python train.py --dataset mnist --type cnn
# Training FeedForward model on KMNIST
RUN python train.py --dataset kmnist --type ff
# Training CNN model on KMNIST
RUN python train.py --dataset kmnist --type cnn

# Default command to run when the container launches.
# This will execute the classification script.
# The actual DATASET and TYPE for classification will be passed
# as environment variables by the Kubernetes Job definition,
# overriding any defaults set via ENV in this Dockerfile (none set here).
CMD ["python", "classify.py"]