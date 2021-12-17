FROM python:3.8.12
COPY ./streamlit_requriements.txt .
RUN pip install -r streamlit_requriements.txt
COPY ./streamlit .
CMD streamlit run app.py --server.port=8502