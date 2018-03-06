import re

source = r"F:\No-Backup Zone\RNN_With_Embeddings\Klassificierade_texter\word2vec.texts_klass.txt"

with open(source) as f:
    contents = f.read()
    count_pos = sum(1 for match in re.finditer(r"\bpositive - \b", contents))
    count_neg = sum(1 for match in re.finditer(r"\bnegative - \b", contents))
    count_obj = sum(1 for match in re.finditer(r"\bobjective - \b", contents))

print("positive: ", count_pos)
print("negative: ", count_neg)
print("objective: ", count_obj)