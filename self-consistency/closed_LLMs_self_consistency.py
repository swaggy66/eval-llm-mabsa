import time
import openai
import json
from transformers import pipeline
import argparse

openai.api_key = "openai key"
openai.api_base = "xxx"

def retry_until_success(func):

    while True:
        try:
            return func()
        except Exception as err:
            print(f"API fail: {err}, retry...")
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
            temperature=args.temp
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

def analyze_sentiments(file_path, output_json_path):
    sentences = read_sentences_from_file(file_path)
    results = []

    for sentence in sentences:
        print('test sentence：' + sentence)
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

Please return the final (not code) output base on the following text in json format.
'''                    f"Text: '{sentence}'."
                ),
            }
        ]

      
        (success, error) = gpt_35_api_stream(messages)
        if success:
        
            generated_text = messages[-1]["content"]
            print(generated_text)
            results.append({
                "sentence": sentence,
                "result": generated_text
            })
        else:
            print(f"error: {error}")

 
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"save in {output_json_path}")


def read_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(" ".join([word.split()[0] for word in sentence]))
                    sentence = []
            else:
                sentence.append(line)
        if sentence:
            sentences.append(" ".join([word.split()[0] for word in sentence]))
    return sentences

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
    parser.add_argument("--test_lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
    parser.add_argument("--temp", default=1.0, type=float, required=True,
                        help="[0.2,.0.4,0.6,0.8,1.0]")

    args = parser.parse_args()


    file_path = f"/data/gold-{args.test_lang}-test.txt"

    output_json_path = f"/data/gemini_self_consistency_results_{args.test_lang}_{args.temp}.json"

    analyze_sentiments(file_path, output_json_path)
