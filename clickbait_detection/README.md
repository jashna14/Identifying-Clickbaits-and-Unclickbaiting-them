# ClickBaitDetection

- Had used 3 datasets Clickbait16k, abcnews and buzzfeed_data(Final_lang.jsonl) for training the model.

- You can reporduce the results by downloading the datasets from [a link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ramaguru_guru_research_iiit_ac_in/EpzKNn4FHytLq-XwXg8YIMsBRodldo19Phs9h1QTFJxS9w?e=fMP1IC) and storing it in `../data` directory.
- To run the code: `python3 train_roberta.py`

- To train the BERT model you need to import BertTokenizer and BertForSequenceClassification and change lines 246 and 334 accordingly.

- To check a title is a clickbaity or not run `python3 script.py` and give clickbait title as input from stdin. 

- Note: That our model has been trained predominantly on abcnews and buzzfeed data.