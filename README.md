# Z-coref: Thai Coreference and Zero Pronoun Resolution

This is a repository for Z-coref project, Thai Coreference and Zero Pronoun Resolution. Please refer to the following paper for more details: [Z-coref: Thai Coreference and Zero Pronoun Resolution]()

Dataset: [Z-coref dataset](https://huggingface.co/datasets/psuwannapich/Z-coref-dataset)

Model: [Z-coref model](https://huggingface.co/psuwannapich/z-coref)

## Installation

1. Clone this repository
2. Install dependencies
```bash
pip install -r requirements.txt
```
## Training

1. Prepare data in JSONL format. Each line is a JSON object with the following keys:
    - `text`/`tokens`: text content of the document. `text` is for raw text, `tokens` is for tokenized text.
    - `clusters`: list of coreference clusters in the format of `[[start, end], [start, end], ...]`
    - `clusters_strings`: list of coreference clusters in string format
2. Train the model using the following command:
```python
from trainer import TrainingArgs, CorefTrainer

args = TrainingArgs(
    model_name_or_path="model_name_or_path"
)

trainer = CorefTrainer(
    args=args,
    train_data_path='path/to/train.jsonl',
    dev_data_path='path/to/dev.jsonl',
    test_data_path='path/to/test.jsonl',
)

trainer.train()
```

Training example can be found in `/examples`.