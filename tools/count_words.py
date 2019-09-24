words = set()
with open('/home/gaoxiao/code/tacotron2/ljs_dataset_folder/metadata.csv', 'r') as f:
    for l in f:
        l = l.split('|')[-1].strip()
        curr = set(l.lower().split())
        words.update(curr)
print(len(words))
