import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import sys
src_text = sys.argv[1]
model_name = 'google/pegasus-xsum'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

f=open("output.txt","w+")
f.write(tgt_text[0])
# assert tgt_text[0] == "California's largest electricity provider has turned off power to hundreds of thousands of customers."