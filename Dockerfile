FROM debian:stable-slim

# Install required packages
RUN apt-get update && apt-get install -y python3 python3-pip

# Install python modules
RUN pip3 install nltk fasttext hdbscan numpy scipy
RUN python3 -c "import nltk; nltk.download('wordnet')"

# Load the fasttext model
COPY model/crawl-300d-2M-subword.bin /model/crawl-300d-2M-subword.bin

# Copy the script
COPY filter.py /filter.py

# Run the script
ENTRYPOINT ["python3", "/filter.py"]
