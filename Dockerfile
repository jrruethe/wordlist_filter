FROM debian:stable-slim

# Install required packages
RUN apt-get update && apt-get install -y python3 python3-pip

# Install python modules
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt --break-system-packages
RUN python3 -c "import nltk; nltk.download('wordnet')"

# Copy the script
COPY filter.py /filter.py

# Run the script
ENTRYPOINT ["python3", "/filter.py"]
