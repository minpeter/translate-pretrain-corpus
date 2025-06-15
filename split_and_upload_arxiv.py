from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv(override=True)

# 1. 데이터셋 로드
dataset = load_dataset("common-pile/arxiv_abstracts_filtered", split="train")

# 2. 5등분
num_splits = 5
split_size = len(dataset) // num_splits
splits = []
for i in range(num_splits):
    start = i * split_size
    end = (i + 1) * split_size if i < num_splits - 1 else len(dataset)
    splits.append(dataset.select(range(start, end)))

# 3. DatasetDict로 묶기
split_names = [f"split_{i+1}" for i in range(num_splits)]
dataset_dict = DatasetDict({name: ds for name, ds in zip(split_names, splits)})

# 4. HuggingFace Hub에 업로드
dataset_dict.push_to_hub("minpeter/arxiv-abstracts-split")
