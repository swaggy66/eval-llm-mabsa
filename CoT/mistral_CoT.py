from transformers import pipeline
import torch
import json
import argparse

local_model_path = "mistralai/Mistral-7B-Instruct-v0.3"

# 加载本地模型
chatbot = pipeline("text-generation", model=local_model_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
parser.add_argument("--test_lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
args = parser.parse_args()

# read  data
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

def analyze_sentiments(file_path, output_file_path):
    sentences = read_sentences_from_file(file_path)
    results = []
    
    for sentence in sentences:
        print('test sentence：' + sentence)
        if args.lang=='en':
            messages = [
                {
                    "role": "user",
                    "content": (
                        # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
                        # "1. Identify the aspect-terms present in the sentence.\n"
                        # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
                        # "3. Provide a reasoning process for how you identified the aspect-terms and assigned their sentiments.\n"
                        # "4. Output the result in the JSON format.\n"
                        # f"Text: '{sentence}'."
                        # "Please identify the aspect-terms mentioned in the following given text and provide a reasoning process for how you identified the aspect-terms and assigned their sentiments. For each aspect-term, determine whether the sentiment is [positive, negative, neutral]. Return the result in a JSON format only with 'aspect' and 'sentiment' as keys."                   
'''
Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each aspect. 
Each entity is associated with a sentiment that can be [positive, negative, or neutral].

Your task is to:

1. Identify the entity with a sentiment mentioned in the given text.If there are no sentiment-bearing entities, the output should be empty.
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
        if args.lang=='fr':
            messages = [
                {
                    "role": "user",
                    "content": (
                        # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
                        # "1. Identify the aspect-terms present in the sentence.\n"
                        # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
                        # "3. Provide a reasoning process for how you identified the aspect-terms and assigned their sentiments.\n"
                        # "4. Output the result in the JSON format.\n"
                        # f"Text: '{sentence}'."
                        "Identifiez les termes d’aspect mentionnés dans le texte donné et fournissez un processus de raisonnement expliquant comment vous avez identifié les termes d’aspect et attribué leurs sentiments. Pour chaque terme d’aspect, déterminez si le sentiment est [positive, negative, neutral]. Retournez le résultat au format JSON uniquement avec les clés 'aspect' et 'sentiment'."                   
                        f"Texte : '{sentence}'."

                    ),
                }
            ]
        if args.lang=='es':
            messages = [
                {
                    "role": "user",
                    "content": (
                        # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
                        # "1. Identify the aspect-terms present in the sentence.\n"
                        # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
                        # "3. Provide a reasoning process for how you identified the aspect-terms and assigned their sentiments.\n"
                        # "4. Output the result in the JSON format.\n"
                        # f"Text: '{sentence}'."
                        "Identifique los términos de aspecto mencionados en el texto dado y proporcione un proceso de razonamiento sobre cómo identificó los términos de aspecto y asignó sus sentimientos. Para cada término de aspecto, determine si el sentimiento es [positive, negative, neutral]. Devuelva el resultado en formato JSON solo con las claves 'aspect' y 'sentimiento'."                   
                        f"Texto: '{sentence}'."

                    ),
                }
            ]
        if args.lang=='nl':
            messages = [
                {
                    "role": "user",
                    "content": (
                        # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
                        # "1. Identify the aspect-terms present in the sentence.\n"
                        # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
                        # "3. Provide a reasoning process for how you identified the aspect-terms and assigned their sentiments.\n"
                        # "4. Output the result in the JSON format.\n"
                        # f"Text: '{sentence}'."
                        "Identificeer de aspecttermen die in de gegeven tekst worden genoemd en geef een redeneerproces over hoe u de aspecttermen hebt geïdentificeerd en hun sentimenten hebt toegewezen. Bepaal voor elke aspectterm of het sentiment [positive, negative, neutral] is. Geef het resultaat terug in JSON-formaat met alleen de sleutels 'aspect' en 'sentiment'."                   
                        f"Tekst: '{sentence}'."

                    ),
                }
            ]
        if args.lang=='ru':
            messages = [
                {
                    "role": "user",
                    "content": (
                        # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
                        # "1. Identify the aspect-terms present in the sentence.\n"
                        # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
                        # "3. Provide a reasoning process for how you identified the aspect-terms and assigned their sentiments.\n"
                        # "4. Output the result in the JSON format.\n"
                        # f"Text: '{sentence}'."
                        "Определите термины аспектов, упомянутые в данном тексте, и предоставьте процесс рассуждений о том, как вы идентифицировали аспекты и назначили им эмоции. Для каждого термина аспекта определите, является ли эмоция [positive, negative, neutral]. Верните результат в формате JSON только с ключами 'aspect' и 'sentiment'."                   
                        f"Текст: '{sentence}'."

                    ),
                }
            ]

        # emply pipeline
        outputs = chatbot(messages, max_new_tokens=512, do_sample=False)

        # extract text
        response_text = outputs[0]["generated_text"][1]["content"]
        print(response_text)
        result = {
            "sentence": sentence,
            "response": response_text
        }
        results.append(result)

    # save as JSON
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# data path 
file_path = f"./dataset/gold-{args.test_lang}-test.txt"
output_file_path = f"./results/mistral_CoT_{args.test_lang}_results.json"
analyze_sentiments(file_path, output_file_path)

# python CoT/mistral_CoT.py  --lang en --test_lang [en,nl,es,ru,fr]
