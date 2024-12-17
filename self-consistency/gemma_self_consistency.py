import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/data/gemma7b_instruct")
model = AutoModelForCausalLM.from_pretrained(
    "/data/gemma7b_instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    revision="float16",
)

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='en', type=str, required=True,
                    help="The language of prompt, selected from: [en, fr, es, nl, ru]")
parser.add_argument("--iterations", default=3, type=int,
                    help="Number of iterations for self-consistency")
args = parser.parse_args()

# Function to read sentences from a txt file, split by empty lines
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
        if sentence:  # Add last sentence if file doesn't end with an empty line
            sentences.append(" ".join([word.split()[0] for word in sentence]))
    return sentences

# Function to generate response using the model
def generate_response(sentence, lang):
    if lang == 'en':
        input_text = (
            "Please identify the aspect-terms mentioned in the following given text. "
            "For each aspect-term, determine whether the sentiment is [positive, negative, neutral]. "
            "Return the result in a JSON format only with 'aspect' and 'sentiment' as keys. "
            f"Text: '{sentence}'."
        )
    # Add other language prompts as necessary
    elif lang == 'fr':
        input_text = (
            "Veuillez identifier les termes d’aspect mentionnés dans le texte donné ci-dessous. "
            "Pour chaque terme d’aspect, déterminez si le sentiment est [positive, negative, neutral]. "
            "Retournez le résultat au format JSON uniquement avec les clés 'aspect' et 'sentiment'. "
            f"Texte : '{sentence}'."
        )
    elif lang == 'es':
        input_text = (
                "Por favor, identifique los términos de aspecto mencionados en el siguiente texto dado. Para cada término de aspecto, determine si el sentimiento es [positive, negative, neutral]. Devuelva el resultado en formato JSON solo con las claves 'aspect' y 'sentimiento'."                   
                f"Texto: '{sentence}'."
        )
    elif lang == 'nl':
        input_text = (
                    "Identificeer de aspect-termen die in de onderstaande gegeven tekst worden genoemd. Bepaal voor elk aspect-term of het sentiment [positive, negative, neutral] is. Geef het resultaat terug in JSON-indeling met alleen de sleutels 'aspect' en 'sentiment'"                   
                f"Tekst: '{sentence}'."
        )
    elif lang == 'ru':
        input_text = (
                    "Пожалуйста, определите аспектные термины, упомянутые в приведённом ниже тексте. Для каждого аспектного термина определите, является ли настроение [positive, negative, neutral]. Верните результат в формате JSON, используя только ключи 'aspect' и 'sentiment'."                   
                f" Текст: '{sentence}'."
        )


    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # Generate output from the model
    outputs = model.generate(**input_ids, max_length=200, do_sample=True)
    
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to analyze sentiments and save results as JSON with self-consistency
def analyze_sentiments(file_path, output_json_path, num_iterations):
    sentences = read_sentences_from_file(file_path)
    results = []

    for sentence in sentences:
        print(f'Testing sentence: {sentence}')
        responses = [generate_response(sentence, args.lang) for _ in range(num_iterations)]

        # Print all generated responses for self-consistency
        print("Responses:")
        for idx, response in enumerate(responses):
            print(f"Iteration {idx+1}: {response}")
        
        results.append({
            "sentence": sentence,
            "responses": responses
        })

    # Save the results as a JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json_path}")

# Specify the txt file path and analyze with self-consistency
file_path = f"/data/gold-{args.lang}-test.txt"
output_json_path = f"/data/gemma_self_consistency_results_{args.lang}.json"

# Run the analysis with self-consistency
analyze_sentiments(file_path, output_json_path, num_iterations=args.iterations)
