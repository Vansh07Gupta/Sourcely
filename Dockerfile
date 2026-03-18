FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY veriGraph.py config.py models.py graph_store.py extractor.py rag_engine.py dashboard.py embeddings.py pdf_ingest.py ./
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "veriGraph.py", "--server.address", "0.0.0.0"]
