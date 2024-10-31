FROM python:3.10
WORKDIR /app
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y libgl1

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose Streamlit port and run the app
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
