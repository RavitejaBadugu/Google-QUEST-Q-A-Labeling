FROM python:3.8.12
COPY ./streamlit_requriements.txt .
RUN pip install -r streamlit_requriements.txt
COPY . .
EXPOSE 8502
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=8502 