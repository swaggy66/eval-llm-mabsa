import argparse
import json
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer, util


local_model_path = "/data/llama3.1"
chatbot = pipeline("text-generation", model=local_model_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")


sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='en', type=str, required=True,
                    help="The language of prompt, selected from: [en, fr, es, nl, ru]")
parser.add_argument("--topk", default=5, type=int, help="Number of top similar sentences to retrieve from training data")
parser.add_argument("--test_lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
args = parser.parse_args()


def read_sentences_with_entities(file_path):
    sentences = []
    current_sentence = []
    entities = []
    current_entity = []
    entity_label = None

    label_mapping = {
        "T-POS": "positive",
        "T-NEG": "negative",
        "T-NEU": "neutral"
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_sentence:  
                    sentences.append((" ".join([word.split()[0] for word in current_sentence]), entities))
                    current_sentence = []
                    entities = []
                    current_entity = []
                continue

            word, label, sentiment = line.split()
            current_sentence.append(line)

            if label != "O":
                if entity_label is None:
                    entity_label = label_mapping.get(label, label)
                    current_entity = [word]
                else:
                    current_entity.append(word)
            else:
                if current_entity:
                    entities.append({"entity": " ".join(current_entity), "sentiment": entity_label})
                    current_entity = []
                    entity_label = None


        if current_sentence:
            sentences.append((" ".join([word.split()[0] for word in current_sentence]), entities))
    
    return sentences



def get_topk_similar_sentences(input_sentence, train_sentences, top_k=5):
    input_embedding = sbert_model.encode(input_sentence, convert_to_tensor=True)
    train_embeddings = [sbert_model.encode(sent[0], convert_to_tensor=True) for sent in train_sentences]
    
    similarities = [util.pytorch_cos_sim(input_embedding, train_emb).item() for train_emb in train_embeddings]
    sorted_sentences = sorted(zip(train_sentences, similarities), key=lambda x: x[1], reverse=True)

    return sorted_sentences[:top_k]

def analyze_sentiments(test_file_path, train_file_path, output_json_path, top_k=5):

    train_sentences_with_entities = read_sentences_with_entities(train_file_path)
    test_sentences = read_sentences_with_entities(test_file_path)
    
    results = []

    for test_sentence, _ in test_sentences:

        topk_sentences_with_entities = get_topk_similar_sentences(test_sentence, train_sentences_with_entities, top_k)


        similar_sentences_context = "\n".join([f"Sentence: {sent[0]}, Entities: {sent[1]}" for sent, _ in topk_sentences_with_entities])

        if args.lang == 'en':
            messages = [
                {
                    "role": "user",
                    "content": (
 '''
Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity.

Each entity is associated with a sentiment that can be [positive, negative, or neutral].

Your task is to:

1. Identify the entity with a sentiment mentioned in the given text.
2. For each identified entity, determine the sentiment in the label set (positive, negative, or neutral).
3. The output should be a list of dictionaries, where each dictionary contains the entity with a sentiment and its corresponding sentiment. If there are no sentiment-bearing entities in the text, the output should be an empty list.

Example Output format:

[
  {"entity": "<entity>", "sentiment": "<label>"}
]
'''
                        f"Here are {top_k} similar sentences from the training set:\n"
                        f"{similar_sentences_context}\n"

                        "Please return the final (not code) output base on the following text in json format."

                        f"Text: '{test_sentence}'\n"
                        # "Please identify the aspect-terms mentioned in the following given text. For each aspect-term, determine whether the sentiment is [positive, negative, neutral]. Return the result in a JSON format only with 'aspect' and 'sentiment' as keys."
                    ),
                }
            ]
        print('111',test_sentence)

        outputs = chatbot(messages, max_new_tokens=512, do_sample=False)
        generated_text = outputs[0]["generated_text"][1]["content"]
        print('333',generated_text)
        results.append({
            "sentence": test_sentence,
            "result": generated_text,
            "topk_similar_sentences": topk_sentences_with_entities
        })


    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"save in {output_json_path}")


train_file_path = f"/data/gold-{args.test_lang}-train.txt"
test_file_path = f"/data/gold-{args.test_lang}-test.txt"
output_json_path = f"/data/result/llama_fewshot_results_{args.test_lang}.json"


analyze_sentiments(test_file_path, train_file_path, output_json_path, top_k=args.topk)
