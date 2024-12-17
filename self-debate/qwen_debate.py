import argparse
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# # Load Qwen2 model and tokenizer
# device = "cuda"  # specify device

# model_name = "Qwen/Qwen2.5-7B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set local model path
model_name = "Qwen/Qwen2.5-7B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="cuda"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load local model
chatbot = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.float32},
    device_map="auto",
)


# Function to read sentences from the txt file
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

# Debate structure functions
def create_initial_message(sentence):
    return [
        {
            "role": "user",
            "content": f'''
Aspect-Based Sentiment Analysis (ABSA) involves identifying specific entity (such as a person, product, service, or experience) mentioned in a text and determining the sentiment expressed toward each entity.

Each entity is associated with a sentiment that can be [positive, negative, or neutral].

Your task is to:

1. Identify the entity with a sentiment mentioned in the given text. If there are no sentiment-bearing entities, the output should be empty.
2. For each identified entity, determine the sentiment in the label set (positive, negative, or neutral).

Example Output format:

[
  {{"entity": "<entity>", "sentiment": "<label>"}}
]

Please return the final (not code) output based on the following text in json format.
Text: '{sentence}'.
'''
        }
    ]

def create_commentary_message(first_response, sentence):
    return [
        {
            "role": "user",
            "content": f'''
The source sentence is: '{sentence}'.

The first response result: {first_response}

Please review and comment on the following response. Provide corrections if necessary or add more details to improve the result (both entity and sentiment).

Example Output format:
[
  {{"entity": "<entity>", "sentiment": "<label>", "review": "<review>"}}
]

Please return the final (not code) output based on the above sentence in json format.
'''
        }
    ]

def create_second_round_message(first_response, first_commentary, sentence):
    return [
        {
            "role": "user",
            "content": f'''
The source sentence is: '{sentence}'.

The first response result: {first_response}.
The first commentary result: {first_commentary}.

Based on the initial response and commentary, please further debate and refine the analysis. If there are any conflicting opinions or uncertainties, resolve them (both entity and sentiment).

Example Output format:
[
  {{"entity": "<entity>", "sentiment": "<label>", "review": "<review>"}}
]

Please return the final (not code) output based on the above sentence in json format.
'''
        }
    ]

# Function to analyze sentiments and save results to JSON
# def analyze_sentiments(file_path, output_json_path):
#     sentences = read_sentences_from_file(file_path)
#     results = []

#     for sentence in sentences:
#         print('Analyzing sentence: ' + sentence)

#         # Round 1: Initial response from the first agent
#         initial_messages = create_initial_message(sentence)
#         text_1 = tokenizer.apply_chat_template(initial_messages, tokenize=False, add_generation_prompt=True)
#         model_inputs_1 = tokenizer([text_1], return_tensors="pt").to(device)
#         generated_ids_1 = model.generate(model_inputs_1.input_ids, max_new_tokens=512, do_sample=False, temperature=0.0)
#         response_1 = tokenizer.batch_decode(generated_ids_1, skip_special_tokens=True)[0]
#         print("Agent 1 Initial Response:\n", response_1)

#         # Round 2: Commentary from the second agent
#         commentary_messages = create_commentary_message(response_1, sentence)
#         text_2 = tokenizer.apply_chat_template(commentary_messages, tokenize=False, add_generation_prompt=True)
#         model_inputs_2 = tokenizer([text_2], return_tensors="pt").to(device)
#         generated_ids_2 = model.generate(model_inputs_2.input_ids, max_new_tokens=512, do_sample=False, temperature=0.0)
#         response_2 = tokenizer.batch_decode(generated_ids_2, skip_special_tokens=True)[0]
#         print("Agent 2 Commentary:\n", response_2)

#         # Round 3: Second-round debate from the first agent
#         second_round_messages = create_second_round_message(response_1, response_2, sentence)
#         text_3 = tokenizer.apply_chat_template(second_round_messages, tokenize=False, add_generation_prompt=True)
#         model_inputs_3 = tokenizer([text_3], return_tensors="pt").to(device)
#         generated_ids_3 = model.generate(model_inputs_3.input_ids, max_new_tokens=512, do_sample=False, temperature=0.0)
#         response_3 = tokenizer.batch_decode(generated_ids_3, skip_special_tokens=True)[0]
#         print("Agent 1 Second Round Response:\n", response_3)

#         # Store all results
#         results.append({
#             "sentence": sentence,
#             "initial_response": response_1,
#             "commentary": response_2,
#             "second_round_response": response_3
#         })

#     # Save results to JSON
#     with open(output_json_path, 'w', encoding='utf-8') as json_file:
#         json.dump(results, json_file, ensure_ascii=False, indent=4)

#     print(f"Results saved to {output_json_path}")

def analyze_sentiments(file_path, output_file_path):
    sentences = read_sentences_from_file(file_path)
    results = []

    for sentence in sentences:
        print('test sentence' + sentence)

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


# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='en', type=str, required=True,
                    help="The language of prompt, selected from: [en, fr, es, nl, ru]")
parser.add_argument("--test_lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
args = parser.parse_args()

# File paths
file_path = f"./dataset/gold-{args.test_lang}-test.txt"
output_file_path = f"./results/qwen_debate_{args.test_lang}_results.json"
analyze_sentiments(file_path, output_file_path)

# python Debate/qwen_debate.py  --lang en --test_lang [en,nl,es,ru,fr]
