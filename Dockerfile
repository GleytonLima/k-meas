FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyYAML (necessário para ler configs)
RUN pip install pyyaml

# Copiar código fonte
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

# Criar diretório de outputs
RUN mkdir -p /app/outputs

# Entrypoint
ENTRYPOINT ["python", "src/run_pipeline.py"]

