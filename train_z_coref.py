from fastcoref import TrainingArgs, CorefTrainer
import spacy
import spacy_pythainlp.core

nlp = spacy.blank("th")
root_dir = "/home/poomphob/Desktop/Thesis/s2e_coref/data/17_10_2023_doccano"
root_dir = "/home/poomphob/Desktop/Thesis/s2e_coref/data/16_01_2024_doccano"


args = TrainingArgs(
    output_dir="xlm-v-base",
    overwrite_output_dir=True,
    model_name_or_path="airesearch/wangchanberta-base-att-spm-uncased",
    # model_name_or_path="xlm-roberta-base",
    # model_name_or_path="bert-base-multilingual-cased",
    # model_name_or_path="clicknext/phayathaibert",
    device="cuda:0",
    epochs=200,
    logging_steps=150,
    eval_steps=100,
    # max_span_length=20,
    dropout_prob=0.4,
    max_segment_len=512,
    # ffnn_size=721,
    # top_lambda=0.6468,
    max_tokens_in_batch=8000,
)

trainer = CorefTrainer(
    args=args,
    nlp=nlp,
    train_file=f"{root_dir}/train_tokens.jsonl",
    dev_file=f"{root_dir}/val_tokens.jsonl",  # optional
    # test_file='/home/poomphob/Desktop/Thesis/fastcoref/test.py'   # optional
)

# trainer = CorefTrainer(
#     args=args,
#     nlp=nlp,
#     train_file="/home/poomphob/Desktop/Thesis/s2e_coref/raw_data/hann_coref/val.json",
#     dev_file="/home/poomphob/Desktop/Thesis/s2e_coref/raw_data/hann_coref/test.json",  # optional
#     # test_file='/home/poomphob/Desktop/Thesis/fastcoref/test.py'   # optional
# )


trainer.train()
