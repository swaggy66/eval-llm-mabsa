import json  
  


f_input_folder="/data/result_new/gpt4o/"
f_input_p="gpt4omini_debate_ru_results.json"
f_input_pred=f"{f_input_folder}{f_input_p}"
f_output_true_folder="/data/result_new/gpt4o_analysis/"
f_output_true=f"/data/gold_test_ru.json"


with open(f_input_pred, 'r') as f:  
    predictions = json.load(f)  
  
with open(f_output_true, 'r') as f:  
    ground_truth = json.load(f)  
  
# label set
label_set = {'positive', 'negative', 'neutral'}  
  

evaluation_results = {}  


turn1={'initial_result':"1", 'explanation_result':"2", 'self_check_result':"3"} 
turn2={'initial_response':"1", 'commentary':"2", 'second_round_response':"3"} 
turn3={'initial_result':"1", 'explanation':"2", 'final_result':"3"} 
turn4={'initial_result':"1", 'explanation_result':"2", 'final_result':"3"} 


possible_keys = ['result', 'initial_response', 'commentary', 'second_round_response']  
# possible_keys = ['result', 'initial_result', 'explanation_result', 'self_check_result']  
# possible_keys = ['result', 'initial_result', 'explanation', 'final_result']  
# possible_keys = ['result', 'initial_result', 'explanation_result', 'final_result']  


for key in possible_keys:  

    TP = 0  
    FP = 0  
    FN = 0  
    #total entity
    len_pred=0
 
    # error type
    error_types = {  
        "OOD": 0,  
        "Wrong": 0,  
        "Contain gold": 0,  
        "Contained by gold": 0,  
        "Overlap with gold": 0,  
        "Completely-O": 0,  
        "OOD mentions": 0 ,
        "Omitted mention":0 
    }  
  

    if key !='result':
        # 逐句对比  
        for pred, truth in zip(predictions, ground_truth):  
            if key in pred:  
                print(pred)
                pred_entities = {(item['entity'], item['sentiment']) for item in pred[key]}  
                truth_entities = {(item['entity'], item['sentiment']) for item in truth['result']}  
                len_pred+=len(pred_entities)
                sentence = pred['sentence']
  
                # TP
                TP += len(pred_entities & truth_entities)
  
                # FP
                FP += len(pred_entities - truth_entities)
  
                # FN
                FN += len(truth_entities - pred_entities)

                for entity, sentiment in pred_entities:
                    # Check for OOD types
                    if sentiment not in label_set:
                        # print('1',entity,sentiment)
                        error_types["OOD"] += 1
                        continue

                    # Check for OOD mentions
                    if entity not in sentence:
                        # print('1',entity,sentiment)
                        # print('2',sentence)
                        error_types["OOD mentions"] += 1
                        continue

                    # Check against ground truth entities
                    matched_gold = [(gold_entity, gold_sentiment) for (gold_entity, gold_sentiment) in truth_entities if gold_entity == entity or gold_entity in entity or entity in gold_entity]

                    if matched_gold:
                        for gold_entity, gold_sentiment in matched_gold:
                            if gold_entity == entity and sentiment != gold_sentiment:
                                error_types["Wrong"] += 1  # Predicted type incorrect but in the given label set
                            
                            # Check for "Contain gold"
                            elif gold_entity in entity and gold_entity != entity and sentiment == gold_sentiment:
                                # print('1',gold_entity,'2',entity)
                                error_types["Contain gold"] += 1  # Predicted mentions containing gold mentions
                            
                            # Check for "Contained by gold"
                            elif entity in gold_entity and gold_entity != entity and sentiment == gold_sentiment:
                                # print('1',gold_entity,'2',entity)                  
                                error_types["Contained by gold"] += 1  # Predicted mentions contained by gold mentions
                            break
                    else:

                        overlap_with_gold = any(not (entity.endswith(gold_entity) or gold_entity.endswith(entity)) for (gold_entity, gold_sentiment) in truth_entities)
                        
                        if entity in sentence and not any(entity in gold_entity for (gold_entity, _) in truth_entities):
                            error_types["Completely-O"] += 1  # Completely-O
                            #print('1',gold_entity,'2',entity)                  
                        # elif overlap_with_gold:
                        else:
                            pass
                            #print('1',gold_entity,'2',entity)                               
                            #error_types["Overlap with gold"] += 1  # Overlap with gold
        error_types["Omitted mention"] = FN
        error_types["Overlap with gold"] += len_pred-error_types["Completely-O"]-error_types["Contained by gold"]-error_types["Contain gold"]-error_types["Wrong"]-error_types["OOD mentions"]-error_types["OOD"]-error_types["Omitted mention"] # Overlap with gold
        if error_types["Overlap with gold"]<0:
            error_types["Overlap with gold"]=0
        print('total entity',len_pred)
        # 计算 Micro-F1  
        Precision = TP / (TP + FP) if TP + FP > 0 else 0  
        Recall = TP / (TP + FN) if TP + FN > 0 else 0  
        Micro_F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall > 0 else 0  
  

        evaluation_results[turn2[key]] = {  
            "TP": TP,  
            "FP": FP,  
            "FN": FN,  
            "Precision": Precision,  
            "Recall": Recall,  
            "Micro_F1": Micro_F1,  
            "Error_Types": error_types  
        }  
  

for key, result in evaluation_results.items():  
    print(f"Results for key '{key}':")  
    print('TP', result['TP'])  
    print('FP', result['FP'])  
    print('FN', result['FN'])  
    print('Precision', result['Precision'])  
    print('Recall', result['Recall'])  
    print("Micro-F1:", result['Micro_F1'])  
    for error_type, count in result['Error_Types'].items():  
        print(f"{error_type}: {count}")  
    print()
# f_save=f"{f_output_true_folder}{f_input_p}_results.json"
print(f"{f_output_true_folder}{f_input_p.split('.')[0]}_analysis.json")
f_save=f"{f_output_true_folder}{f_input_p.split('.')[0]}_analysis.json"

# save in JSON  
with open(f_save, 'w', encoding='utf-8') as f:  
    json.dump(evaluation_results, f, ensure_ascii=False, indent=4)