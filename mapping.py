import os
import json
import re
import argparse
#from data.load_dataset import read_file

def ord_tags(text):
    s_to_remove = ['[TRIPLE]', '[/TRIPLE]', '/TRIPLE','TRIPLE', '[', ']', ',', '.', "'", '"']
    for punctuation in s_to_remove:
        text = text.replace(punctuation, '')  
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)  
    text = text.split('#newline#')[0]
    return text.strip()

def struct_tags(contents):
    replacements = {
        '[ SNT)': '[SNT]', '[ /SNT)': '[/SNT]', '[SNT)': '[SNT]', '[/SNT)': '[/SNT]',
        '[ SNT] ': '[SNT]', '[/ SNT] ': '[/SNT]', '[ Snt] ': '[SNT]', '[/ Snt] ': '[/SNT]',
        '[sNT)': '[SNT]', '[/sNT)': '[/SNT]', '[T] ': '[SNT]', '[/T] ': '[/SNT]',
        '[SNT?]': '[SNT]', '[/SNT?]': '[/SNT]', '[SOD]': '[SNT]', '[/SOD]': '[/SNT]',
        ' SNT]': '[SNT]', ' /SNT]': '[/SNT]', '[SENT]': '[SNT]', '[/SENT]': '[/SNT]',
        '[S NT]': '[SNT]', '[/S NT]': '[/SNT]', '[SVP]': '[SNT]', '[/SVP]': '[/SNT]',
        '[SDP]': '[SNT]', '[/SDP]': '[/SNT]', '[SNT...]': '[SNT]', '[/SNT...]': '[/SNT]',
        '[SOTT]': '[SNT]', '[/SOTT]': '[/SNT]', '[sNT]': '[SNT]', '[/sNT]': '[/SNT]',
        '[SOT]': '[SNT]', '[/SOT]': '[/SNT]', '[SNS]': '[SNT]', '[/SNS]': '[/SNT]',
        '], [': '] [', ']. [': '] [', '],':']', '].':']', '[[[': '[', '[[': '[',
        '[SNT].': '[SNT]', '[/SNT].': '[SNT]', '(SNT)':'[SNT]', '(/SNT)':'[/SNT]',
        '[SNT],':'[SNT]', '[/SNT],':'[/SNT]', '[SNT].':'[SNT]', '[/SNT].':'[/SNT]',
        '[SNA]':'[SNT]', '[/SNA]': '[/SNT]', '[SNS]':'[SNT]', '[/SNS]':'[/SNT]',
        "[SNT']":'[SNT]', "[/SNT']":'[/SNT]', '[SS]':'[SNT]', '[/SS]':'[/SNT]',
        '[SUNT]':'[SNT]', '[/SUNT]':'[/SNT]', '[DNT|':'[SNT]', '[/DNT|':'[/SNT]',
        '[SENT]':'[SNT]', '[/SENT]':'[/SNT]','[SNOR]':'[SNT]', '[/SNOR]':'[/SNT]',

        }

    xcontents = contents
    for old, new in replacements.items():
        xcontents = xcontents.replace(old, new)

    pattern = re.compile(r'\[\s*(\/?\w+)\s*\]')
    pattern2 = re.compile(r'\[\s*(\/?\s*[\w\.,]+)\s*\]\s*[\.,]?\s*')

    cleaned_text = re.sub(pattern, lambda m: f"[{'/' if m.group(1).startswith('/') else ''}SNT] ", xcontents)
    cleaned_text = re.sub(pattern2, lambda m: f"[{'/' if m.group(1).startswith('/') else ''}SNT] ", cleaned_text)
    cleaned_text = re.sub(r'\[SNT\]([^\s])', r'[SNT] \1', cleaned_text)  # Add space after [SNT] if not followed by a space
    cleaned_text = cleaned_text.replace('  ', ' ')
    # Split the text into lines
    #cleaned_text_lines = cleaned_text.strip().split('\n')

    return cleaned_text


def lex_tags(contents):
    replacements = {
        'ENTITITY':'ENTITY', 'ENTITIES':'ENTITY', 'ENTID':'ENTITY', 'ENTITY/':'ENTITY-', ' E-':' ENTITY-',
        'ENTITYy':'ENTITY', 'ENTITY-"':'ENTITY-', 'AGENT':'PATIENT', 'ENTITY -':'ENTITY-', 'PATIENT -':'PATIENT-',
        'ENTITY]':'ENTITY', 'ENTIRE': 'ENTITY', 'ENTITY)':'ENTITY', 'ENITITE-':'ENTITY-', 'ENTIT-':'ENTITY-',
        'ENTIER-':'ENTITY-', 'ENTRY-':'ENTITY-', 'ENTIENT-':'ENTITY-', 'ENTITY CITITY': 'ENTITY',
        ' ia ':' is a ', 'active andperson':'active, person',
        ' [':'[', '.[':'[','[ ':'[', ' ]':']', 'ENTITY- ':'ENTITY-',
        'PATIENT':'ENTITY',"<":"", ">":"", "#":"",
        }
    xcontents = contents
    for old, new in replacements.items():
        xcontents = xcontents.replace(old, new)

    pattern = r'\b(ENTITY)[\'"/\]?\s*[-_]\s*(\d+)(?:[.,]?\s*)\b'
    xcontents = re.sub(pattern, r'\1-\2 ', xcontents, flags=re.IGNORECASE)
    pattern2 = r'\b(ENTITY)(\d+)\b'
    xcontents = re.sub(pattern2, r'\1-\2', xcontents, flags=re.IGNORECASE)
    pattern3 = r'\b(ENT[A-Z]\w+)-(\d+)\b'
    xcontents = re.sub(pattern3, r'ENTITY-\2', xcontents, flags=re.IGNORECASE)
    xcontents = xcontents.replace("'", " '")
    #return xcontents
    # Splitting text into sentences
    if ": " in xcontents:
        xcontents = xcontents.split(": ")[-1]
    sentences = xcontents.split(". ")
    if sentences[-1].endswith("?"):
        sentences.pop()  # Remove the last sentence if it ends with a question mark
    # Reconstructing the text
    summary = ". ".join(sentences)
    return summary.strip()
    

# Read data from files
def read_file_map(path, task):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            if task == "structuring":
                lines = [struct_tags(line.strip()) for line in contents.split('\n')]
                #print(lines)
            elif task == "lexicalization":
                lines = [lex_tags(line.strip()) for line in contents.split('\n')]
            else:
                lines = [ord_tags(line.strip()) for line in contents.split('\n')]
                
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines

def read_file(path):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            contents = contents.replace('<', '[').replace('>', ']')#.replace('"',"^")
            lines = [line.strip() for line in contents.split('\n')]
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


#def lexout2regin(lex_out, triples):
    #entities = entity_mapping(triples)
    #print(f"Entites: {entities}")
    #for i, w in enumerate(lex_out):
        #cleaned_word = w.strip(".,!\"'")
        #if cleaned_word.startswith("ENTITY-") and cleaned_word.strip() in entities:
            #print( f"{i}: {cleaned_word}=={entities[cleaned_word]}")
            #lex_out[i] = entities[cleaned_word] + w[len(cleaned_word):]
    #return ' '.join(lex_out).replace(" '", "'")#.replace("^",'"')

def lexout2regin(lex_out, triples):
    entities = entity_mapping(triples)
    #print(f"Entities: {entities}")
    for i, w in enumerate(lex_out):
        cleaned_word = w.strip(".,!\"'")
        # Check if the word matches the pattern ENTITY-\d+
        if re.match(r'^ENTITY-\d+$', cleaned_word):
            # Check if the cleaned word exists in the entities mapping
            if cleaned_word in entities:
                # Replace the entity with its corresponding value, preserving quotation marks if present
                #print( f"{i}: {cleaned_word}=={entities[cleaned_word]}")
                original_word = lex_out[i]
                replacement = entities[cleaned_word].strip('"')
                lex_out[i] = original_word.replace(cleaned_word, replacement)
    return ' '.join(lex_out).replace(" '", "'")


def run(out_path, entries_path, pre_task):
    outputs = [out.split() for out in read_file_map(out_path, pre_task)]
    entries = [split_triples(y.split()) for y in read_file(entries_path)]
    
    #if len(outputs) != len(entries):
        #print(f"Current {pre_task} task length {len(outputs)}", f"Previous task input length {len(entries)}")
        #raise ValueError("Number of outputs does not match number of entries")
    #start_index = max(0, len(outputs) - len(entries))
    for i, entry in enumerate(entries):
        #i = start_index + i
        if pre_task == "ordering":
            yield orderout2structin(ordering_out=outputs[i], triples=entry)
        elif pre_task == "structuring":
            yield structout2lexin(struct_out=outputs[i], triples=entry)
        elif pre_task == "lexicalization":
            yield lexout2regin(lex_out=outputs[i], triples=entry)
        else:
            raise ValueError("Invalid pre_task value")



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
