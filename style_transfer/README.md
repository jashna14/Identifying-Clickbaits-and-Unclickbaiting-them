# Style Transfer 

- For training the model run `python3 main.py -datapath ./data/clickbaits/`

- In `data/clickbaits` we have 6 files X.merge(titles) X.labels(corresponding title belongs to style 1 or 0). You can create your own dataset accordingly and give the path. 

- You can create the vocab file needed to run the above code by running `python3 make_vocab.py -file_merge ./data/clickbaits/train.merge -file_vocab ./data/clickbaits/vocab`

- A model will be stored as a checkpoint in `./Checkpoint_3` folder.

- To run on test sentences `python3 main.py -if_eval True -file_save output.txt -checkpoint ./Checkpoint_3/filename -datapath ./data/clickbaits/ -batch_size 1` 


- Please install dependencies from requirements.txt `pip install -r requirements.txt`

[Reference](https://github.com/YouzhiTian/Structured-Content-Preservation-for-Unsupervised-Text-Style-Transfer)