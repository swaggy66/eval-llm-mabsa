#error analysis
import json
f_input_folder="/data/result/zephyr/"
f_input_p="zephyr_consistency_results_nl.json"
f_input_pred=f"{f_input_folder}{f_input_p}"
f_output_true_folder="/data/result/zephyr_analysis/"
f_output_true=f"/data/gold_test_nl.json"

with open(f_input_pred, 'r') as f:
    predictions = json.load(f)

with open(f_output_true, 'r') as f:
    ground_truth = json.load(f)

# label set
label_set = {'positive', 'negative', 'neutral'}


TP = 0
FP = 0
FN = 0


error_types = {
    "OOD": 0,
    "Wrong": 0,
    "Contain gold": 0,
    "Contained by gold": 0,
    "Overlap with gold": 0,
    "Completely-O": 0,
    "OOD mentions": 0
}
len_pred=0

for pred, truth in zip(predictions, ground_truth):
    print(pred)
    pred_entities = {(item['entity'], item['sentiment']) for item in pred['result']}#response
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
                #print(entity,gold_entity)
                error_types["Completely-O"] += 1  # Completely-O
                #print('1',gold_entity,'2',entity)
            else:
                pass
            # elif overlap_with_gold:
                #print('1',gold_entity,'2',entity)
                # error_types["Overlap with gold"] += 1  # Overlap with gold
error_types["Omitted mention"] = FN
error_types["Overlap with gold"] = len_pred-error_types["Completely-O"]-error_types["Contained by gold"]-error_types["Contain gold"]-error_types["Wrong"]-error_types["OOD mentions"]-error_types["OOD"]-error_types["Omitted mention"] # Overlap with gold
if error_types["Overlap with gold"]<0:
    error_types["Overlap with gold"]=0
  
# 计算 Micro-F1
if TP + FP == 0:
    Precision = 0
else:
    Precision = TP / (TP + FP)

if TP + FN == 0:
    Recall = 0
else:
    Recall = TP / (TP + FN)

if Precision + Recall == 0:
    Micro_F1 = 0
else:
    Micro_F1 = 2 * (Precision * Recall) / (Precision + Recall)


print('TP', TP)
print('FP', FP)
print('FN', FN)
print('Precision', Precision)
print('Recall', Recall)
print("Micro-F1:", Micro_F1)


for error_type, count in error_types.items():
    print(f"{error_type}: {count}")

# save as JSON
results = {
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "Precision": Precision,
    "Recall": Recall,
    "Micro_F1": Micro_F1,
    "Error_Types": error_types
}
print(f"{f_output_true_folder}{f_input_p.split('.')[0]}_analysis.json")
f_save=f"{f_output_true_folder}{f_input_p.split('.')[0]}_analysis.json"
with open(f_save, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

