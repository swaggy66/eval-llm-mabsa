from transformers import pipeline
import torch
import json
import argparse


local_model_path = "google/gemma-2-9b-it"


chatbot = pipeline("text-generation", model=local_model_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='en', type=str, required=True,
                    help="The language of prompt, selected from: [en, fr, es, nl, ru]")
parser.add_argument("--test_lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
args = parser.parse_args()


def create_initial_message(sentence):
    return [
        {
            "role": "user",
            "content": (
                # "1. Please identify the aspect-terms mentioned in the sentence.\n"
                # "2. For each aspect-term, determine whether the sentiment is [positive, negative, neutral].\n"
                # "3. Return the result in a JSON format with 'aspect' and 'sentiment' as keys.\n"
                  '''
Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity. 
Each entity is associated with a sentiment that can be [positive, negative, or neutral].

Your task is to:

1. Identify the entity with a sentiment mentioned in the given text.If there are no sentiment-bearing entities, the output should be empty.
2. For each identified entity, determine the sentiment in the label set (positive, negative, or neutral).

Example Output format:

[
  {"entity": "<entity>", "sentiment": "<label>"}
]

Please return the final (not code) output base on the following text in json format.
'''
                f"Text: '{sentence}'."
            ),
        }
    ]

def create_commentary_message(previous_response, sentence):
    return [
        {
            "role": "user",
            "content": (
                # f"Task:Please identify the entities mentioned in the sentence.For each aspect-term, determine whether the sentiment is [positive, negative, neutral].Return the result in a JSON format with 'aspect' and 'sentiment' as keys."
                '''
                Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity. 
                Each entity is associated with a sentiment that can be [positive, negative, or neutral].
                '''
                f"The source sentence is: '{sentence}'. The first response result: {previous_response}\n"
                "Please review and comment on the following response. Provide corrections if necessary or add more details to improve the result(both entity and review).\n"
                # "Commentary:"
                '''
                Example Output format:
                Round 1
                [
                {"entity": "<entity>", "sentiment": "<label>","Review":"<review>"}
                ]

                Please return the final (not code) output base on the above sentence in json format.

                '''
            ),
        }
    ]

def create_second_round_message(first_response, first_commentary, sentence):
    return [
        {
            "role": "user",
            "content": (
                # f"Task:Please identify the aspect-terms mentioned in the sentence.For each aspect-term, determine whether the sentiment is [positive, negative, neutral].Return the result in a JSON format with 'aspect' and 'sentiment' as keys."             
                '''
                Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity. 
                Each entity is associated with a sentiment that can be [positive, negative, or neutral].
                '''

                f"The source sentence is: '{sentence}'. The first response result: {first_response}.\n"
                f"The first commentary result: {first_commentary}.\n"
                "Based on the initial response and commentary, please further debate and refine the analysis. If there are any conflicting opinions or uncertainties, resolve them (both entity and review) and provide a more detailed and accurate response.\n"
                # "Second Round Response:"
                '''
                Example Output format:
                Round 2
                [
                {"entity": "<entity>", "sentiment": "<label>","Review":"<review>"}
                ]

                Please return the final (not code) output base on the above sentence in json format.

                '''

            ),
        }
    ]

def analyze_sentiments(file_path, output_file_path):
    sentences = read_sentences_from_file(file_path)
    results = []

    for sentence in sentences:
        print('test sentence：' + sentence)

        # Agent 1: Generate initial response
        initial_messages = create_initial_message(sentence)
        outputs_1 = chatbot(initial_messages, max_new_tokens=512, do_sample=False)
        response_1 = outputs_1[0]["generated_text"][1]["content"]
        print("Agent 1 Response:\n", response_1)

        # Agent 2: Comment on Agent 1's response
        commentary_messages = create_commentary_message(response_1, sentence)
        outputs_2 = chatbot(commentary_messages, max_new_tokens=512, do_sample=False)
        response_2 = outputs_2[0]["generated_text"][1]["content"]
        print("Agent 2 Commentary:\n", response_2)

        # Agent 1: Second round debate based on the first response and commentary
        second_round_messages = create_second_round_message(response_1, response_2, sentence)
        outputs_3 = chatbot(second_round_messages, max_new_tokens=512, do_sample=False)
        response_3 = outputs_3[0]["generated_text"][1]["content"]
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


file_path = f"./dataset/gold-{args.test_lang}-test.txt"
output_file_path = f"./results/gemma_debate_{args.test_lang}_results.json"
analyze_sentiments(file_path, output_file_path)

# python Debate/gemma_debate.py  --lang en --test_lang [en,nl,es,ru,fr]