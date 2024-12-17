import argparse
import json
from transformers import AutoTokenizer, pipeline
import torch

# Set local model path for Zephyr
model_path = "/data/zephyr"

# Load Zephyr model with tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# chatbot = pipeline(
#     "text-generation",
#     model=model_path,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )
chatbot = pipeline("text-generation", model=model_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

# Argument parsing for language selection
parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='en', type=str, required=True,
                    help="The language of prompt, selected from: [en, fr, es, nl, ru]")
parser.add_argument("--test_lang", default='en', type=str, required=True,
                    help="The language of prompt, selected from: [en, fr, es, nl, ru]")
args = parser.parse_args()

# Function to read txt file and split into sentences
def read_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:  # If sentence is not empty
                    sentences.append(" ".join([word.split()[0] for word in sentence]))
                    sentence = []
            else:
                sentence.append(line)
        if sentence:  # Handle last sentence
            sentences.append(" ".join([word.split()[0] for word in sentence]))
    return sentences

# Function to generate recursive prompts
def generate_recursive_prompts(sentence, initial_result=None, explanation_result=None):
    if args.lang == 'en':
        initial_prompt = [{
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
'''
            f"Text: '{sentence}'."),
        }]

        explanation_prompt = [{
                                "role": "user",
                    "content": (
            '''
            Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity. 
            Each entity is associated with a sentiment that can be [positive, negative, or neutral].
            '''     
            f"Here is the sentence: '{sentence}'. You have classified the sentiment of the entities in this sentence. "
            f"Here is your initial result: {initial_result}. Please explain why you classified them in this way. "
           '''
            Example Output format:
            [
            {"entity": "<entity>", "sentiment": "<label>", "Explanation": "<reasoning process>"}
            ]

            '''
            "Please return the final (not code) output base on the above sentence in json format."
        ),}]

        self_check_prompt = [{
                                "role": "user",
                    "content": (
            '''
            Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity. 
            Each entity is associated with a sentiment that can be [positive, negative, or neutral].
            '''     
            f"Here is the sentence: '{sentence}'. You have given a sentiment classification for the entities in this sentence. "
            f"Here is your initial result: {initial_result}. Here is your explanation: {explanation_result}. "
            "Please recheck your classification and explanation. If you find any errors or better classifications, please update your response accordingly. "
            '''
            Example Output format:
            [
            {"entity": "<entity>", "sentiment": "<label>", "Explanation": "<reasoning process>"}
            ]
          
            '''
            "Please return the final (not code) output base on the above sentence in json format."
        ),}]
    # You can expand the above sections for other languages as in your original template.
    return initial_prompt, explanation_prompt, self_check_prompt

# Function to perform sentiment analysis with self-improvement
def analyze_sentiments(file_path, output_json_path):
    sentences = read_sentences_from_file(file_path)
    results = []

    for sentence in sentences:
        print(f'Analyzing sentence: {sentence}')
        
        # Initial result
        initial_prompt, explanation_prompt, self_check_prompt = generate_recursive_prompts(sentence)
        
        # Step 1: Generate initial result
        outputs_initial = chatbot.tokenizer.apply_chat_template(initial_prompt, max_new_tokens=512, tokenize=False, do_sample=False, add_generation_prompt=True)
        initial_result = outputs_initial[0]["generated_text"][1]["content"]
        initial_result = initial_result[len(initial_prompt):]
        # Step 2: Generate explanation based on initial result
        explanation_prompt = generate_recursive_prompts(sentence, initial_result=initial_result)[1]
        outputs_explanation = chatbot.tokenizer.apply_chat_template(explanation_prompt, max_new_tokens=512, tokenize=False, do_sample=False, add_generation_prompt=True)
        explanation_result = outputs_explanation[0]["generated_text"][1]["content"]
        explanation_result = explanation_result[len(explanation_prompt):]
        # Step 3: Self-check and final review
        self_check_prompt = generate_recursive_prompts(sentence, initial_result=initial_result, explanation_result=explanation_result)[2]
        outputs_self_check = chatbot.tokenizer.apply_chat_template(self_check_prompt, max_new_tokens=512, tokenize=False, do_sample=False, add_generation_prompt=True)
        final_result = outputs_self_check[0]["generated_text"][1]["content"]
        final_result = final_result[len(self_check_prompt):]
        print(f"Final result: {final_result}")
        results.append({
            "sentence": sentence,
            "initial_result": initial_result,
            "explanation": explanation_result,
            "final_result": final_result
        })

    # Save results to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json_path}")

# Specify file path and run analysis
file_path = f"/data/gold-{args.test_lang}-test.txt"
output_json_path = f"/data/zephyr_self_improvement_results_{args.test_lang}.json"

analyze_sentiments(file_path, output_json_path)
