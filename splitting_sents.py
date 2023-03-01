import os
import pathlib
from tqdm import tqdm
import spacy

os.chdir(pathlib.Path(__file__).parent.resolve())

nlp = spacy.load("zh_core_web_md")

raw_files = ["raw1.txt", "raw2.txt", "raw3.txt"]

for i, raw_file in enumerate(raw_files):
    sent_file = raw_file.replace("raw", "sent")
    print(f"Processing {raw_file} -> {sent_file} ({i+1}/{len(raw_files)})...")
    with open(raw_file, "r", encoding="utf8") as f_in:
        raw = f_in.read()
        sents = []
        for para in tqdm(raw.split("\n")):
            sents_in_this_para = []
            flag = False
            for i in nlp(para).sents:
                text = i.text.strip()
                if text != "":
                    if (
                        flag
                        or text.startswith("”")
                        or text.startswith("》")
                        or text.startswith("、")
                        or text.startswith("：")
                    ):
                        if sents_in_this_para:
                            sents_in_this_para[-1] = sents_in_this_para[-1] + text
                        else:
                            sents_in_this_para.append(text)
                        if len(text) > 3:
                            flag = False
                    elif len(text) > 3:
                        sents_in_this_para.append(text)
                    else:
                        if sents_in_this_para:
                            sents_in_this_para[-1] = sents_in_this_para[-1] + text
                        else:
                            sents_in_this_para.append(text)
                        flag = True
            sents.extend(sents_in_this_para)
        with open(sent_file, "w", encoding="utf8") as f_out:
            f_out.write("\n".join(sents))

# Note: for better quality, search with the regex expressions like "[\u4e00-\u9fa5]\n" and manually check the matches.
