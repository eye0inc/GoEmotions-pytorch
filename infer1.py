from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint

# m = "monologg/bert-base-cased-goemotions-original"
# m = "ckpt/original/bert-base-cased-goemotions-original/checkpoint-1000"
# m = "ckpt/original/bert-base-cased-goemotions-original/checkpoint-2000"
# m = "ckpt/original/bert-base-cased-goemotions-original/checkpoint-3000"
# m = "ckpt/original/bert-base-cased-goemotions-original/checkpoint-5000"
m = "ckpt/original/bert-base-cased-goemotions-original/checkpoint-20000"
# m = "ckpt/original/bert-base-cased-goemotions-original/checkpoint-27000"

tokenizer = BertTokenizer.from_pretrained(m)
model = BertForMultiLabelClassification.from_pretrained(m)

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)

texts = [
    "Hey that's a thought! Maybe we need [NAME] to be the celebrity vaccine endorsement!",
    "it’s happened before?! love my hometown of beautiful new ken 😂😂",
    "I love you, brother.",
    "Troll, bro. They know they're saying stupid shit. The motherfucker does nothing but stink up libertarian subs talking shit",
]

texts = [
    "I'm furious that maniac stole my shirt, but I love how it looks!",
    "I wonder why he did it?",
    "It's hard to say but I suspect the answer miight be ridiculous."
]

pprint(goemotions(texts))

"""
Output
 [{'labels': ['neutral'], 'scores': [0.9750906]},
 {'labels': ['curiosity', 'love'], 'scores': [0.9694574, 0.9227462]},
 {'labels': ['love'], 'scores': [0.993483]},
 {'labels': ['anger'], 'scores': [0.99225825]}]
"""