import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
device = "cuda"  # specify device

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

# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='en', type=str, required=True,
                    help="The language of the prompt, selected from: [en, fr, es, nl, ru]")
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
        initial_prompt = (
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
            f"Text: '{sentence}'."
        )

        explanation_prompt = (
            # f"Here is the sentence: '{sentence}'. You have classified the sentiment of the aspect-terms in this sentence. "
            # f"Here is your initial result: {initial_result}. Please explain why you classified them this way. "
            # "Include details on the words or phrases that influenced your sentiment classification. Output the explanation in JSON format."
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
        )

        self_check_prompt = (
            # f"Here is the sentence: '{sentence}'. You have provided a sentiment classification for the aspect-terms. "
            # f"Here is your initial result: {initial_result}. Here is your explanation: {explanation_result}. "
            # "Please recheck your classification and explanation. If you find any errors or better classifications, update your response. "
            # "Output the updated classification and explanation in JSON format."
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
        )

    elif args.lang == 'fr':
        initial_prompt = (
            f"Veuillez identifier les termes d’aspect mentionnés dans le texte ci-dessous. "
            f"Pour chaque terme d’aspect, déterminez si le sentiment est [positive, negative, neutral]. "
            f"Retournez le résultat au format JSON avec les clés 'aspect' et 'sentiment'. "
            f"Texte : '{sentence}'."
        )

        explanation_prompt = (
            f"Voici la phrase : '{sentence}'. Vous avez classé le sentiment des termes d'aspect. "
            f"Voici votre résultat initial : {initial_result}. Veuillez expliquer pourquoi vous les avez classés de cette manière. "
            "Incluez des détails sur les mots ou phrases ayant influencé votre classification de sentiment. Fournissez l'explication en JSON."
        )

        self_check_prompt = (
            f"Voici la phrase : '{sentence}'. Vous avez donné une classification des sentiments pour les termes d'aspect. "
            f"Voici votre résultat initial : {initial_result}. Voici votre explication : {explanation_result}. "
            "Veuillez vérifier votre classification et explication. Si vous trouvez des erreurs, veuillez mettre à jour la réponse. "
            "Sortez la classification mise à jour en JSON."
        )

    elif args.lang == 'es':
        initial_prompt = (
            f"Por favor, identifique los términos de aspecto en el siguiente texto. "
            f"Para cada término de aspecto, determine si el sentimiento es [positive, negative, neutral]. "
            f"Devuelva el resultado en formato JSON con las claves 'aspect' y 'sentimiento'. "
            f"Texto: '{sentence}'."
        )

        explanation_prompt = (
            f"Aquí está la oración: '{sentence}'. Has clasificado el sentimiento de los términos de aspecto. "
            f"Resultado inicial: {initial_result}. Por favor, explica por qué clasificaste de esta manera. "
            "Incluye detalles sobre las palabras o frases que influyeron en tu clasificación. Proporciona la explicación en JSON."
        )

        self_check_prompt = (
            f"Aquí está la oración: '{sentence}'. Has proporcionado una clasificación de sentimiento para los términos de aspecto. "
            f"Resultado inicial: {initial_result}. Explicación: {explanation_result}. "
            "Por favor, vuelva a verificar su clasificación. Si encuentra errores, actualice su respuesta. "
            "Devuelva la clasificación y explicación actualizadas en formato JSON."
        )

    elif args.lang == 'nl':
        initial_prompt = (
            f"Identificeer de aspect-termen in de onderstaande tekst. "
            f"Bepaal voor elk aspect-term of het sentiment [positive, negative, neutral] is. "
            f"Geef het resultaat in JSON-indeling met de sleutels 'aspect' en 'sentiment'. "
            f"Tekst: '{sentence}'."
        )

        explanation_prompt = (
            f"Hier is de zin: '{sentence}'. Je hebt het sentiment van de aspect-termen geclassificeerd. "
            f"Initiële resultaat: {initial_result}. Leg uit waarom je deze classificatie hebt gemaakt. "
            "Voeg details toe over de woorden of zinnen die jouw classificatie beïnvloedden. Geef de uitleg in JSON-indeling."
        )

        self_check_prompt = (
            f"Hier is de zin: '{sentence}'. U hebt een classificatie van het sentiment voor de aspecttermen gegeven. "
            f"Initiële resultaat: {initial_result}. Uitleg: {explanation_result}. "
            "Controleer de classificatie opnieuw. Als u fouten vindt, werk de reactie bij. "
            "Geef de bijgewerkte classificatie en uitleg in JSON-indeling."
        )

    elif args.lang == 'ru':
        initial_prompt = (
            f"Пожалуйста, определите аспектные термины в тексте ниже. "
            f"Для каждого термина определите, является ли настроение [positive, negative, neutral]. "
            f"Верните результат в формате JSON с ключами 'aspect' и 'sentiment'. "
            f"Текст: '{sentence}'."
        )

        explanation_prompt = (
            f"Вот предложение: '{sentence}'. Вы классифицировали настроение аспектных терминов. "
            f"Начальный результат: {initial_result}. Объясните, почему вы их так классифицировали. "
            "Включите детали слов или фраз, которые повлияли на вашу классификацию. Ответьте в формате JSON."
        )

        self_check_prompt = (
            f"Вот предложение: '{sentence}'. Вы дали классификацию настроения для аспектных терминов. "
            f"Начальный результат: {initial_result}. Объяснение: {explanation_result}. "
            "Проверьте свою классификацию снова. Если вы найдете ошибки, обновите свой ответ. "
            "Верните обновленную классификацию и объяснение в формате JSON."
        )

    return initial_prompt, explanation_prompt, self_check_prompt


# Sentiment analysis function that applies recursive prompts
def analyze_sentiments(file_path, output_json_path):
    sentences = read_sentences_from_file(file_path)
    results = []

    for sentence in sentences:
        print(f"Analyzing sentence: {sentence}")

        # Generate initial prompt
        initial_prompt, explanation_prompt, self_check_prompt = generate_recursive_prompts(sentence)


        initial_prompt, explanation_prompt, self_check_prompt = generate_recursive_prompts(sentence)
        messages = [{"role": "user", "content": initial_prompt}]
        outputs = chatbot(messages, max_new_tokens=512, do_sample=False)
        initial_result = outputs[0]["generated_text"][1]["content"]
        print("initial_result：", initial_result)

  
        explanation_prompt_with_result = generate_recursive_prompts(sentence, initial_result)[1]
        explanation_messages = [{"role": "user", "content": explanation_prompt_with_result}]
        explanation_outputs = chatbot(explanation_messages, max_new_tokens=512, do_sample=False)
        explanation_result = explanation_outputs[0]["generated_text"][1]["content"]
        print("explanation_result：", explanation_result)

  
        self_check_prompt_with_result = generate_recursive_prompts(sentence, initial_result, explanation_result)[2]
        self_check_messages = [{"role": "user", "content": self_check_prompt_with_result}]
        self_check_outputs = chatbot(self_check_messages, max_new_tokens=512, do_sample=False)
        self_check_result = self_check_outputs[0]["generated_text"][1]["content"]
        print("self_check_result：", self_check_result)

        # Store the result
        results.append({
            "sentence": sentence,
            "initial_result": initial_result,
            "explanation_result": explanation_result,
            "final_result": self_check_result
        })

    # Save results to JSON
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json_path}")


# Define input and output file paths
file_path = f"./dataset/gold-{args.test_lang}-test.txt"
output_file = f"./results/qwen_self_inprove_{args.test_lang}_results.json"


analyze_sentiments(file_path, output_file)

# python Self-Improvement/qwen_self_improve.py  --lang en --test_lang [en,nl,es,ru,fr]
