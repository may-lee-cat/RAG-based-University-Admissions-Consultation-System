# A University Admissions Consultaion System Based on RAG

## Prepare

1. Insure your Python version is above 3.8
2. Install all the packages from `requirements.txt`
3. Add your own data in `data`. Nested folders are allowed

## Run

Running `main.py` when interacting in command line. If running for the first time, alter the value of `if_data_init` to `True` in line `75`. Running `app.py` when interacting through a Web UI inferface. If running for the first time, 
alter the value of `if_data_init` to `True` in line `42`. System will generate index files `models` and `bm25_retriever.pkl` automatically.

If not running for the first time, alter the value of `if_data_init` to `False`. System will read local index files rather than generating them again. This change will save some time.

## Interact

Taking UI interface interacion as an example. After running `app.py`, a web interface will pop up. Input your queries in the dialog box below, and then click "Send".

<img width="773" height="716" alt="image" src="https://github.com/user-attachments/assets/06fdb128-0b87-4559-85b2-482919be78aa" />

You can view the RRF results and retrieved data through the backend log. In addition, you can also view the complete information returned by the LLM API, including the number of tokens and other information.

<img width="945" height="279" alt="image" src="https://github.com/user-attachments/assets/d18397ec-bb8d-48e9-9af6-dd837d71c36e" />
