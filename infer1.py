from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

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
    "I'm furious that bitch stole my shirt, but I love her for it!"
]

pprint(goemotions(texts))

"""
Output
 [{'labels': ['neutral'], 'scores': [0.9750906]},
 {'labels': ['curiosity', 'love'], 'scores': [0.9694574, 0.9227462]},
 {'labels': ['love'], 'scores': [0.993483]},
 {'labels': ['anger'], 'scores': [0.99225825]}]
"""