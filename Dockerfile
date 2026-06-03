FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/default-java \
    PATH="/usr/lib/jvm/default-java/bin:$PATH"

# Install default JRE (headless) for Spark and tools for build/downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Symlink Java to a standard dynamic location
RUN ln -s $(dirname $(dirname $(readlink -f $(which java)))) /usr/lib/jvm/default-java

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "dashboards/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]