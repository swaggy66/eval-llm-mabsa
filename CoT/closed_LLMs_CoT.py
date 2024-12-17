import time
import openai
import json
from transformers import pipeline
import argparse


openai.api_key = "openai key"
openai.api_base = "xxxxxx"

def retry_until_success(func):

    while True:
        try:
            return func()
        except Exception as err:
            print(f"API connection failed: {err}, trying...")
            time.sleep(5)  # wait 5 seconds

def gpt_35_api_stream(messages: list):
    """

    Args:
        messages (list)

    Returns:
        tuple: (results, error_desc)
    """
    '''[claude-3-haiku-20240307,gpt-4o-mini,gemini-1.5-flash-latest]'''
    def api_call():
        response = openai.ChatCompletion.create(
            model='claude-3-haiku-20240307',
            messages=messages,
            stream=True,
            temperature=0
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                print(f'completed data: {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                completion[delta_k] += delta_v
        messages.append(completion)
        return (True, '')

    # retry
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
Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each aspect. 
Each entity is associated with a sentiment that can be [positive, negative, or neutral].

Your task is to:

1. Identify the entity with a sentiment mentioned in the given text.If there are no sentiment-bearing entities, the output should be empty list.
2. For each identified entity, determine the sentiment in the label set (positive, negative, or neutral).
3. Provide a reasoning process for how you identified the entities and assigned their sentiments.

Example Output format:
json
[
  {"entity": "<entity>", "sentiment": "<label>",, "Explanation": "<reasoning process>"}
]

Please return the final (not code) output base on the following text in json format.
'''
                f"Text: '{sentence}'."
                ),
            }
        ]

        # retry
        (success, error) = gpt_35_api_stream(messages)
        if success:
            # 提取生成的结果并保存
            generated_text = messages[-1]["content"]
            print(generated_text)
            results.append({
                "sentence": sentence,
                "result": generated_text
            })
        else:
            print(f"faile: {error}")


    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"result save in {output_json_path}")


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
    args = parser.parse_args()

 
    file_path = f"/data/gold-{args.test_lang}-test.txt"
    output_json_path = f"/data/result/claude3_CoT/claude3_CoT_results_{args.test_lang}.json"

    analyze_sentiments(file_path, output_json_path)
