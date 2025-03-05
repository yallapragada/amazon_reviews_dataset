FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Set environment variable for Hugging Face cache directory
ENV HF_HOME=/downloads/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV LOCAL_RANK=1

# Create the Hugging Face cache directory
RUN mkdir -p $HF_HOME/accelerate
RUN mkdir -p $HF_HOME/hub

# Create folder for output data
RUN mkdir -p /app/data/output

# Copy the requirements file and training script
COPY requirements.txt /app/
COPY Appliances.csv /app/data/
COPY llm_fine_tuning_config.txt /app/
COPY llm_finetune.py /app/
COPY linux_hardware_monitoring.py /app/
COPY default_config.yaml $HF_HOME/accelerate/

# RUN apt update && apt install -y \
#     build-essential \
#     libpq-dev \
#     gcc \
#     nvidia-cuda-toolkit \
#     && rm -rf /var/lib/apt/lists/*

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    build-essential \
    libpq-dev \
    gcc \
    nvidia-cuda-toolkit \
    tzdata && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the container entrypoint
# ENTRYPOINT ["accelerate", "launch", "/app/llm_finetune.py", "--data_dir", "/app/data", "--output_dir", "/app/data/output", "--model_name", "mistralai/Mistral-7B-v0.3", "--load_in_4bit"]
ENTRYPOINT ["accelerate", "launch", "/app/llm_finetune.py"]