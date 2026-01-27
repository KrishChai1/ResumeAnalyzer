FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY intelligent_parser.py .
COPY resume_parser_mcp.py .
COPY api_server.py .

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "api_server.py"]
