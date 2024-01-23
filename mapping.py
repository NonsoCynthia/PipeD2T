import os
import json
import re
import argparse
from data.load_dataset import read_file


# Read data from files
def read_file_map(path):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            # Do not replace `<` and `>` if they are not part of your tags
            # contents = contents.replace('<', '[').replace('>', ']')
            xcontents = contents.replace('[ /SNT)', '[/SNT]').replace('[/SNT)', '[/SNT]').replace('[/ SNT] ', '[/SNT]')
            xcontents = xcontents.replace('[/ Snt] ', '[/SNT]').replace('[/sNT)', '[/SNT]').replace('[/T] ', '[/SNT]')
            xcontents = xcontents.replace('[SNT?]', '[SNT]').replace('/SOD', '/SNT').replace('[ SNT]', '[SNT]')
            xcontents = xcontents.replace('/SNT]', '[/SNT]')
            xcontents = xcontents.replace('[[[', '[').replace('[[', '[')        #.replace('[[/TRIPLE]', '[/TRIPLE]')
            # xcontents = xcontents.replace('<', '[').replace('>', ']')
            lines = [line.strip() for line in xcontents.split('\n')]
            # print(lines)
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines
        
def prcs_entry(entry):
    cleaned_word = entry.strip('.,!')
    if cleaned_word.startswith("ENTITY-"):
        entry = cleaned_word.replace('.', ' .').replace(',', ' ,').replace('!', ' !').strip('.')
    return entry

def delist(lst):
  lst = ' '.join(lst).strip()
  return lst

def write_file(write_path, result):
    with open(write_path, 'w') as f:
        f.write('\n'.join(result))


def split_triples(text):
    triples, triple = [], []
    for w in text:
        if w not in ['[TRIPLE]', '[/TRIPLE]']:
            triple.append(w)
        elif w == '[/TRIPLE]':
            triples.append(triple)
            triple = []
    return triples


def join_triples(triples):
    result = []
    for triple in triples:
        result.append('[TRIPLE]')
        result.extend(triple)
        result.append('[/TRIPLE]')
    return result

def join_struct(sentences):
    result = []
    for sentence in sentences:
        result.append('[SNT]')
        for triple in sentence:
            result.append('[TRIPLE]')
            result.extend(triple)
            result.append('[/TRIPLE]')
        result.append('[/SNT]')
    return result

def entity_mapping(triples):
    entitytag = {}
    entities = {}
    entity_pos = 1
    for triple in triples:
        agent = triple[0]
        if agent not in entitytag:
            entitytag[agent] = 'ENTITY-' + str(entity_pos)
            entities['ENTITY-' + str(entity_pos)] = agent
            entity_pos += 1

        patient = triple[-1]
        if patient not in entitytag:
            entitytag[patient] = 'ENTITY-' + str(entity_pos)
            entities['ENTITY-' + str(entity_pos)] = patient
            entity_pos += 1
    return entities


def orderout2structin(ordering_out, triples):
    ord_triples = []
    if len(triples) == 1:
        ord_triples.extend(triples)
    else:
        added = []
        for predicate in ordering_out:
            for i, triple in enumerate(triples):
                if predicate.strip() == triple[1].strip() and i not in added:
                    ord_triples.append(triple)
                    added.append(i)
                    break
    return ' '.join(join_triples(ord_triples))


def structout2lexin(struct_out, triples):
    sentences, snt = [], []
    for w in struct_out:
        if w.strip() not in ['[SNT]', '[/SNT]']:
            snt.append(w.strip())

        if w.strip() == '[/SNT]':
            sentences.append(snt)
            snt = []

    struct, struct_unit = [], []
    if len(triples) == 1:
        struct.append(triples)
    else:
        added = []
        for snt in sentences:
            for predicate in snt:
                for i, triple in enumerate(triples):
                    if predicate.strip() == triple[1].strip() and i not in added:
                        struct_unit.append(triple)
                        added.append(i)
                        break
            struct.append(struct_unit)
            struct_unit = []
    return ' '.join(join_struct(struct))



# def lexout2regin(lex_out, triples):
#     entities = entity_mapping(triples)
#     for i, w in enumerate(lex_out):
#         w = w.strip().strip('.').strip(',').strip('!')
#         if w.strip() in entities:
#             lex_out[i] = entities[w.strip()]
#     return ' '.join(lex_out)

def lexout2regin(lex_out, triples):
    entities = entity_mapping(triples)
    for i, w in enumerate(lex_out):
        # Identify entities and replace them with their values
        cleaned_word = w.strip('.,!')
        if cleaned_word.startswith("ENTITY-") and cleaned_word in entities:
            lex_out[i] = entities[cleaned_word] + w[len(cleaned_word):]

    return ' '.join(lex_out)


def run(out_path, entries_path, pre_task):
    outputs = [out.split() for out in read_file_map(out_path)]
    entries = [split_triples(y.split()) for y in read_file(entries_path)]
    # with open(out_path) as f:
    #     outputs = f.read().split('\n')
    # outputs = [out.split() for out in outputs]
    # with open(entries_path) as f:
    #     entries = f.read().split('\n')
    # print(entries)
    for i, entry in enumerate(entries):
        if pre_task == "ordering":
            yield orderout2structin(ordering_out=outputs[i], triples=entry)
        elif pre_task == "structuring":
            yield structout2lexin(struct_out=outputs[i], triples=entry)
        # elif pre_task == "lexicalization":
        else:
            yield lexout2regin(lex_out=outputs[i], triples=entry)


def main(pre_data, pipe_data, pre_task, model):

    data_suffix = {
        "ordering": "eval",
        "structuring": "ordering",
        "lexicalization": "structuring",
        "reg": "lex",
        "sr": "reg"
    }.get(pre_task, "reg")

    if pre_task == "ordering":
        data_path_dev = os.path.join(pre_data, f"results/dev.eval") #results/dev.ordering
        data_path_test = os.path.join(pre_data, f"results/test.eval") #results/test.ordering

        # out_path_dev = os.path.join(pipe_data, f"{pre_task}/{model}/{pre_task}_pipeline_eval.txt") #results/ordering/t5/ordering_pipeline_eval.txt
        # out_path_test = os.path.join(pipe_data, f"{pre_task}/{model}/{pre_task}_pipeline_test.txt") #results/ordering/t5/ordering_pipeline_test.txt
    elif pre_task == "lexicalization":
        data_path_dev = os.path.join(pipe_data, f"ordering/{model}/dev.ordering.mapped") #results/ordering/t5/dev.ordering.mapped
        data_path_test = os.path.join(pipe_data, f"ordering/{model}/test.ordering.mapped") #results/ordering/t5/test.ordering.mapped
 
    else:
        data_path_dev = os.path.join(pipe_data, f"{data_suffix}/{model}/dev.{data_suffix}.mapped") #results/ordering/t5/dev.ordering.mapped
        data_path_test = os.path.join(pipe_data, f"{data_suffix}/{model}/test.{data_suffix}.mapped") #results/ordering/t5/test.ordering.mapped

    out_path_dev = os.path.join(pipe_data, f"{pre_task}/{model}/{pre_task}_pipeline_eval.txt") #results/structuring/t5/structuring_pipeline_eval.txt
    out_path_test = os.path.join(pipe_data, f"{pre_task}/{model}/{pre_task}_pipeline_test.txt")#results/structuring/t5/structuring_pipeline_test.txt
    # print(out_path_dev)
    # print(out_path_test)

    write_path_dev = os.path.join(pipe_data, f"{pre_task}/{model}/dev.{pre_task}.mapped") #results/structuring/t5/dev.structuring.mapped
    write_path_test = os.path.join(pipe_data, f"{pre_task}/{model}/test.{pre_task}.mapped")#results/structuring/t5/test.structuring.mapped
    # print(write_path_dev)
    # print(write_path_test)

    result_dev = run(out_path=out_path_dev, entries_path=data_path_dev, pre_task=pre_task)
    result_test = run(out_path=out_path_test, entries_path=data_path_test, pre_task=pre_task)

    print(f"Mapping generated dev to the original file for {pre_task}-{model}")
    write_file(write_path_dev, result_dev)
    
    print(f"Mapping generated test to the original file for {pre_task}-{model}")
    write_file(write_path_test, result_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--previous_data", help="path to the previous task generations")
    parser.add_argument("--pipeline_data", help="path to the pipeline data")
    parser.add_argument("--previous_task", help="Previous task pipeline")
    parser.add_argument("--Gen_model", help="Model used")
    args = parser.parse_args()

    # args.previous_data = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/data/deepnlg"
    # args.pipeline_data = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results"
    # args.previous_task = "ordering" 
    # args.Gen_model = "t5"
    # next_task = "structuring"
    # first_task = "ordering"

    main(pre_data=args.previous_data, pipe_data=args.pipeline_data, pre_task=args.previous_task, model=args.Gen_model)
