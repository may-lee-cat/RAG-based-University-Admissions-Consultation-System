# A University Admissions Consultaion System Based on RAG

## Prepare

1. Insure your Python version is above 3.8
2. Install all the packages from `requirements.txt`
3. Add your own data in `data`. Nested folders are allowed

## Run

Running `main.py` when interacting in command line. If running for the first time, alter the value of `if_data_init` to `True` in line `75`. Running `app.py` when interacting through a Web UI inferface. If running for the first time, 
alter the value of `if_data_init` to `True` in line `42`. System will generate index files `models` and `bm25_retriever.pkl` automatically.

If not running for the first time, alter the value of `if_data_init` to `False`. System will read local index files rather than generating them again. This change will save some time.
