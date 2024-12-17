import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

local_model_path = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    #attn_implementation="flash_attention_2"
)

chatbot = pipeline("text-generation", model=model,tokenizer=tokenizer)

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
parser.add_argument("--test_lang", default='en', type=str, required=True,
                        help="The language of prompt, selected from: [en, fr, es, nl, ru]")
args = parser.parse_args()


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


def generate_recursive_prompts(sentence, initial_result=None, explanation_result=None):

    if args.lang=='en':
        initial_prompt = (
            # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
            # "1. Identify the aspect-terms present in the sentence.\n"
            # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
            # "3. Output the result in JSON format.\n"
            # f"Text: '{sentence}'."
            # "Please identify the aspect-terms mentioned in the following given text. For each aspect-term, determine whether the sentiment is [positive, negative, neutral]. Return the result in a JSON format only with 'aspect' and 'sentiment' as keys."                   
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
            # f"Here is your initial result: {initial_result}. Please explain why you classified them in this way. "
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
            # f"Here is the sentence: '{sentence}'. You have given a sentiment classification for the aspect-terms in this sentence. "
            # f"Here is your initial result: {initial_result}. Here is your explanation: {explanation_result}. "
            # "Please recheck your classification and explanation. If you find any errors or better classifications, please update your response accordingly. "
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
    if args.lang=='fr':
        initial_prompt = (
            # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
            # "1. Identify the aspect-terms present in the sentence.\n"
            # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
            # "3. Output the result in JSON format.\n"
            # f"Text: '{sentence}'."
            "Veuillez identifier les termes d’aspect mentionnés dans le texte donné ci-dessous. Pour chaque terme d’aspect, déterminez si le sentiment est [positive, negative, neutral]. Retournez le résultat au format JSON uniquement avec les clés 'aspect' et 'sentiment'."                   
            f"Texte : '{sentence}'."

        )

    
        explanation_prompt = (
            f"Voici la phrase : '{sentence}'. Vous avez classé le sentiment des termes d'aspect dans cette phrase. "
            f"Voici votre résultat initial : {initial_result}. Veuillez expliquer pourquoi vous les avez classés de cette manière. "
            "Incluez des détails sur les mots ou phrases qui ont influencé votre classification de sentiment. Fournissez l'explication au format JSON."
        )
        
       
        self_check_prompt = (
            f"Voici la phrase : '{sentence}'. Vous avez donné une classification des sentiments pour les termes d'aspect dans cette phrase. "
            f"Voici votre résultat initial : {initial_result}. Voici votre explication : {explanation_result}. "
            "Veuillez revérifier votre classification et votre explication. Si vous trouvez des erreurs ou des classifications meilleures, veuillez mettre à jour votre réponse en conséquence. "
            "Sortez la classification et l'explication mises à jour au format JSON."
        )
    if args.lang=='es':
        initial_prompt = (
            # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
            # "1. Identify the aspect-terms present in the sentence.\n"
            # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
            # "3. Output the result in JSON format.\n"
            # f"Text: '{sentence}'."
            "Por favor, identifique los términos de aspecto mencionados en el siguiente texto dado. Para cada término de aspecto, determine si el sentimiento es [positive, negative, neutral]. Devuelva el resultado en formato JSON solo con las claves 'aspect' y 'sentimiento'."                   
            f"Texto: '{sentence}'."

        )

 
        explanation_prompt = (
            f"Aquí está la oración: '{sentence}'. Has clasificado el sentimiento de los términos de aspecto en esta oración. "
            f"Aquí está tu resultado inicial: {initial_result}. Por favor, explica por qué los clasificaste de esta manera. "
            "Incluye detalles sobre las palabras o frases que influyeron en tu clasificación de sentimiento. Proporciona la explicación en formato JSON."
        )
        
  
        self_check_prompt = (
            f"Aquí está la oración: '{sentence}'. Usted ha dado una clasificación de sentimiento para los términos de aspecto en esta oración. "
            f"Aquí está su resultado inicial: {initial_result}. Aquí está su explicación: {explanation_result}. "
            "Por favor, vuelva a verificar su clasificación y explicación. Si encuentra errores o mejores clasificaciones, actualice su respuesta en consecuencia."
            "Salga la clasificación y la explicación actualizadas en formato JSON."
        )
    if args.lang=='nl':
        initial_prompt = (
            # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
            # "1. Identify the aspect-terms present in the sentence.\n"
            # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
            # "3. Output the result in JSON format.\n"
            # f"Text: '{sentence}'."
            "Identificeer de aspect-termen die in de onderstaande gegeven tekst worden genoemd. Bepaal voor elk aspect-term of het sentiment [positive, negative, neutral] is. Geef het resultaat terug in JSON-indeling met alleen de sleutels 'aspect' en 'sentiment'."                   
            f"Tekst: '{sentence}'."

        )

   
        explanation_prompt = (
            f"Hier is de zin: '{sentence}'. Je hebt het sentiment van de aspect-termen in deze zin geclassificeerd. "
            f"Hier is je initiële resultaat: {initial_result}. Leg alsjeblieft uit waarom je ze op deze manier hebt geclassificeerd. "
            "Voeg details toe over de woorden of zinnen die je sentimentclassificatie hebben beïnvloed. Geef de uitleg in JSON-formaat."
        )
        
    
        self_check_prompt = (
            f"Hier is de zin: '{sentence}'. U hebt een sentimentclassificatie gegeven voor de aspecttermen in deze zin."
            f"Hier is uw initiële resultaat: {initial_result}. Hier is uw uitleg: {explanation_result}."
            "Controleer alstublieft uw classificatie en uitleg opnieuw. Als u fouten of betere classificaties vindt, werk dan uw antwoord dienovereenkomstig bij."
            "Geef de bijgewerkte classificatie en uitleg in JSON-formaat."
        )
    if args.lang=='ru':
        initial_prompt = (
            # "Given a sentence and an aspect-term label set {positive, negative, neutral}, your task is to:\n"
            # "1. Identify the aspect-terms present in the sentence.\n"
            # "2. For each identified aspect-term, assign one of the following labels: [positive, negative, neutral].\n"
            # "3. Output the result in JSON format.\n"
            # f"Text: '{sentence}'."
            "Пожалуйста, определите аспектные термины, упомянутые в приведённом ниже тексте. Для каждого аспектного термина определите, является ли настроение [positive, negative, neutral]. Верните результат в формате JSON, используя только ключи 'aspect' и 'sentiment'."                   
            f"Текст: '{sentence}'."

        )

        
        explanation_prompt = (
            f"Вот предложение: '{sentence}'. Вы классифицировали тональность аспектных терминов в этом предложении. "
            f"Вот ваш первоначальный результат: {initial_result}. Пожалуйста, объясните, почему вы классифицировали их таким образом. "
            "Укажите детали слов или фраз, которые повлияли на вашу классификацию тональности. Введите объяснение в формате JSON."
        )
        
    
        self_check_prompt = (
            f"Вот предложение: '{sentence}'. Вы дали классификацию тональности для терминов аспектов в этом предложении. "
            f"Вот ваш первоначальный результат: {initial_result}. Вот ваше объяснение: {explanation_result}."
            "Пожалуйста, пересмотрите вашу классификацию и объяснение. Если вы найдете ошибки или лучшие классификации, обновите ваш ответ соответствующим образом."
            "Выведите обновленную классификацию и объяснение в формате JSON."
        )
    
    return initial_prompt, explanation_prompt, self_check_prompt


def analyze_sentiments(file_path, output_file):
    sentences = read_sentences_from_file(file_path)
    results = []
    
    for i, sentence in tqdm(enumerate(sentences)):
        print(f"test {i + 1}: {sentence}")
        
      
        initial_prompt, explanation_prompt, self_check_prompt = generate_recursive_prompts(sentence)
        messages = [{"role": "user", "content": initial_prompt}]
        outputs = chatbot(messages, max_new_tokens=512, do_sample=False, return_full_text=False,temperature=0.0)
        initial_result = outputs[0]["generated_text"]#[1]["content"]
        print("initial_result：", initial_result)

 
        explanation_prompt_with_result = generate_recursive_prompts(sentence, initial_result)[1]
        explanation_messages = [{"role": "user", "content": explanation_prompt_with_result}]
        explanation_outputs = chatbot(explanation_messages, max_new_tokens=512, do_sample=False, return_full_text=False,temperature=0.0)
        explanation_result = explanation_outputs[0]["generated_text"]#[1]["content"]
        print("explanation_result：", explanation_result)

        
        self_check_prompt_with_result = generate_recursive_prompts(sentence, initial_result, explanation_result)[2]
        self_check_messages = [{"role": "user", "content": self_check_prompt_with_result}]
        self_check_outputs = chatbot(self_check_messages, max_new_tokens=512, do_sample=False, return_full_text=False,temperature=0.0)
        self_check_result = self_check_outputs[0]["generated_text"]#[1]["content"]
        print("self_check_result：", self_check_result)
        
      
        result = {
            "sentence": sentence,
            "initial_result": initial_result,
            "explanation_result": explanation_result,
            "self_check_result": self_check_result
        }
        results.append(result)
    
 
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"save {output_file}")


file_path = f"./dataset/gold-{args.test_lang}-test.txt"
output_file = f"./results/phi_self_inprove_{args.test_lang}_results.json"


analyze_sentiments(file_path, output_file)

# python Self-Improvement/phi_self_improve.py  --lang en --test_lang [en,nl,es,ru,fr]