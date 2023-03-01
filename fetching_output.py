import csv
import os
import pathlib
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.chdir(pathlib.Path(__file__).parent.resolve())

tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-large-chinese", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("BAAI/glm-large-chinese", trust_remote_code=True).to(
    "cuda"
)

sent_files = ["sent1.txt", "sent2.txt", "sent3.txt"]

for i, sent_file in enumerate(sent_files):
    logit_file = sent_file.replace("sent", "logit").replace(".txt", ".csv")
    token_file = sent_file.replace("sent", "token")
    print(f"Processing {sent_file} -> {logit_file} ({i+1}/{len(sent_files)})...")
    with open(sent_file, "r", encoding="utf8") as f_in, open(
        logit_file, "w", encoding="utf8"
    ) as f_out:
        sents = f_in.read().split("\n")
        sent_max_len = max(len(sent) for sent in sents)
        writer = csv.writer(f_out)
        for sent in tqdm(sents):
            encodings = tokenizer(sent, return_tensors="pt")
            seq_len = encodings.input_ids.size(1)
            for end_loc in range(2, seq_len):
                input_ids = encodings.input_ids[:, :end_loc].to("cuda")
                target_ids = input_ids.clone()
                target_ids[:, 0 : end_loc - 1] = -100
                with torch.no_grad():
                    writer.writerow(
                        [
                            tokenizer.convert_ids_to_tokens(target_ids[0, -1].item()),
                            model(
                                input_ids,
                                labels=target_ids,
                                output_attentions=True,
                            )
                            .logits[0, -1, target_ids[0, -1].item()]
                            .item(),
                        ]
                    )
