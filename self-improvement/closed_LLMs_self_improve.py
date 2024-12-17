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


def create_initial_message(sentence):
    return [
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
''' + f"Text: '{sentence}'."
            ),
        }
    ]


def create_commentary_message(previous_response, sentence):
    return [
        {
            "role": "user",
            "content": (
            '''
            Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity. 
            Each entity is associated with a sentiment that can be [positive, negative, or neutral].
            '''     
            f"Here is the sentence: '{sentence}'. You have classified the sentiment of the entities in this sentence. "
            f"Here is your initial result: {previous_response}. Please explain why you classified them in this way. "
           '''
            Example Output format:
            [
            {"entity": "<entity>", "sentiment": "<label>", "Explanation": "<reasoning process>"}
            ]

            '''
            "Please return the final (not code) output base on the above sentence in json format."

            ),
        }
    ]


def create_second_round_message(first_response, first_commentary, sentence):
    return [
        {
            "role": "user",
            "content": (
            '''
            Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity. 
            Each entity is associated with a sentiment that can be [positive, negative, or neutral].
            '''     
            f"Here is the sentence: '{sentence}'. You have given a sentiment classification for the entities in this sentence. "
            f"Here is your initial result: {first_response}. Here is your explanation: {first_commentary}. "
            "Please recheck your classification and explanation. If you find any errors or better classifications, please update your response accordingly. "
            '''
            Example Output format:
            [
            {"entity": "<entity>", "sentiment": "<label>", "Explanation": "<reasoning process>"}
            ]
          
            '''
            "Please return the final (not code) output base on the above sentence in json format."

            ),
        }
    ]


def analyze_sentiments(file_path, output_file_path):
    sentences = read_sentences_from_file(file_path)
    results = []

    for sentence in sentences:
        print('test sentence' + sentence)

        # Agent 1: 
        initial_messages = create_initial_message(sentence)
        (success, error) = gpt_35_api_stream(initial_messages)
        response_1 = initial_messages[-1]["content"]
        print("Agent 1 Response:\n", response_1)

        # Agent 2:
        commentary_messages = create_commentary_message(response_1, sentence)
        (success, error) = gpt_35_api_stream(commentary_messages)
        response_2 = commentary_messages[-1]["content"]
        print("Agent 2 Commentary:\n", response_2)

        # Agent 1: 
        second_round_messages = create_second_round_message(response_1, response_2, sentence)
        (success, error) = gpt_35_api_stream(second_round_messages)
        response_3 = second_round_messages[-1]["content"]
        print("Agent 1 Second Round Response:\n", response_3)

     
        result = {
            "sentence": sentence,
            "initial_response": response_1,
            "commentary": response_2,
            "second_round_response": response_3
        }
        results.append(result)

   
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


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
    output_file_path = f"/data/gemini_self_improve_{args.test_lang}_results.json"
    
    analyze_sentiments(file_path, output_file_path)
