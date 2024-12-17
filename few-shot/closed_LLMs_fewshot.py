import time
import openai
import json
from transformers import pipeline
import argparse
from sentence_transformers import SentenceTransformer, util


openai.api_key = "openai key"
openai.api_base = "xxx"

# 加载SBERT模型
sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1',device=1)

def retry_until_success(func):

    while True:
        try:
            return func()
        except Exception as err:

            time.sleep(5)  

def gpt_35_api_stream(messages: list):
    """

    Args:
        messages (list):

    Returns:
        tuple: (results, error_desc)
    """
    '''[claude-3-haiku-20240307,gpt-4o-mini,gemini-1.5-flash-latest]'''

    def api_call():
        response = openai.ChatCompletion.create(
            model='gemini-1.5-flash-latest',
            messages=messages,
            stream=True,
            temperature=0
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            # if event['choices'][0]['finish_reason'] == 'stop':
            if 'finish_reason' in event['choices'][0] and event['choices'][0]['finish_reason'] == 'stop':
                print(f'data: {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                completion[delta_k] += delta_v
        messages.append(completion)
        return (True, '')

    return retry_until_success(api_call)

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

def get_topk_similar_sentences(input_sentence, train_sentences, top_k=1):
    input_embedding = sbert_model.encode(input_sentence, convert_to_tensor=True)
    train_embeddings = [sbert_model.encode(sent[0], convert_to_tensor=True) for sent in train_sentences]
    
    similarities = [util.pytorch_cos_sim(input_embedding, train_emb).item() for train_emb in train_embeddings]
    sorted_sentences = sorted(zip(train_sentences, similarities), key=lambda x: x[1], reverse=True)

    return sorted_sentences[:top_k]

def analyze_sentiments(test_file_path, train_file_path, output_json_path, top_k=1):
    train_sentences_with_entities = read_sentences_with_entities(train_file_path)
    test_sentences = read_sentences_with_entities(test_file_path)
    
    results = []

    for test_sentence, _ in test_sentences:
        topk_sentences_with_entities = get_topk_similar_sentences(test_sentence, train_sentences_with_entities, top_k)

        similar_sentences_context = "\n".join([f"Sentence: {sent[0]}, Entities: {sent[1]}" for sent, _ in topk_sentences_with_entities])

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
f"Here are {top_k} similar sentences from the training set:{similar_sentences_context}"
'''  
Please return the final (not code) output based on the following text in json format.
'''                    f"Text: '{test_sentence}'."
                ),
            }
        ]
        print('111',test_sentence)
        print('222',similar_sentences_context)
        (success, error) = gpt_35_api_stream(messages)
        print('333',messages)
        if success:
            generated_text = messages[-1]["content"]
            print(generated_text)
            results.append({
                "sentence": test_sentence,
                "result": generated_text,
                "topk_similar_sentences": topk_sentences_with_entities
            })
        else:
            print(f"fail: {error}")

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"save in  {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
    parser.add_argument("--topk", default=1, type=int, help="Number of top similar sentences to retrieve from training data,selected from [1,2,4,8,16]")
    parser.add_argument("--test_lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
    args = parser.parse_args()

    train_file_path = f"/data/gold-{args.test_lang}-train.txt"
    test_file_path = f"/data/gold-{args.test_lang}-test.txt"
    output_json_path = f"/data/result/gemini_fewshot/gemini_fewshot_results_{args.test_lang}_{args.topk}.json"

    analyze_sentiments(test_file_path, train_file_path, output_json_path, top_k=args.topk)
