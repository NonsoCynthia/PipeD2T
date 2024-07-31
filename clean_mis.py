import os 
import re

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
        '], [': '] [', ']. [': '] [', '],':']', '].':']', '[[[': '[',
        '[[': '[', '>':']', '<':'[', ' ] ':' ', ' [ ':' ',
        '[/SNT] [/SNT]':'[/SNT]',
        '[SNT].': '[SNT]', '[/SNT].': '[SNT]', '(SNT)':'[SNT]', '(/SNT)':'[/SNT]',
        '[SNT],':'[SNT]', '[/SNT],':'[/SNT]', '[SNT].':'[SNT]', '[/SNT].':'[/SNT]',
        '[SNA]':'[SNT]', '[/SNA]': '[/SNT]', '[SNS]':'[SNT]', '[/SNS]':'[/SNT]',
        "[SNT']":'[SNT]', "[/SNT']":'[/SNT]', '[SS]':'[SNT]', '[/SS]':'[/SNT]',
        '[SUNT]':'[SNT]', '[/SUNT]':'[/SNT]', '[DNT|':'[SNT]', '[/DNT|':'[/SNT]',
        '[SENT]':'[SNT]', '[/SENT]':'[/SNT]','[SNOR]':'[SNT]', '[/SNOR]':'[/SNT]',
        '[ /SNT]':'[/SNT]', '[ SNT]':'[SNT]', '/SNT]':'[/SNT]',
        '[[/SNT]':'[/SNT]', '[  SRI]':'[SNT]', 'SRI]':'[SNT]',
        '[/SNT] [/SNT] [SNT]':'[/SNT] [SNT]', '[/SNT] [SNT] [SNT]':'[/SNT] [SNT]'
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

def process_str_data(data):
    patn = r'(?<=\[SNT\])\s*(.*?)\s*(?=\[SNT\])'
    patn1 = r'(\[SNT\][^\[]+)\s+(\1)+'
    for _ in range(2):
        data = [struct_tags(text).split('Input:')[0].strip() for text in data]
        data = [text.split('Output:')[0].replace(' ]', '').strip() for text in data]
        data = [text + ' [/SNT]' if not text.endswith('[/SNT]') else text for text in data]
        data = [re.sub(patn, r' \1 [/SNT] ', text).replace(' [/ ', ' ').replace('[/SNT] [/SNT]', '[/SNT]').replace('[SNT] [SNT]', '[SNT]').replace(' [/ ', ' ') for text in data]
        data = [re.sub(patn1, r'\1', text) for text in data]
    return data


def process_lex_data(data):
    data = [text.split('  ')[0] for text in data]
    data = [text.replace('[SNT] [TRIPLE] ', '').strip() for text in data]
    data = [text.split('Input:')[0].strip() for text in data]
    data = [text.split('Output:')[-1].strip() for text in data]
    data = [text.replace('[ ', '').replace('  ]', '').strip() for text in data]
    data = [re.sub(r'\s+', ' ', text).strip() for text in data]
    data = [text.replace('[TRIPLE]', '').replace('--', '').replace('- ', '').strip() for text in data]
    return data

def process_e2e_data(data):
    data = [text.split('Input:')[0].strip() for text in data]
    data = [text.split('Output:')[-1].strip() for text in data]
    data = [text.split(' [  ')[-1].replace('  ]','').strip() for text in data]
    data = [re.sub(r'\s+', ' ', text).strip() for text in data]
    return data

def process_mis_str(data):
    replc = {'[/SNT]': '', 'SNT]': '', '[/':'', '[':'', ']':'', '/':'', '[SNT]':'', '/SNT]':'' }
    data = [text.split('[  Input:')[0].strip() for text in data]
    data = [text.split('Input:')[0].strip() for text in data]
    data = [text.split('Output:')[-1].strip() for text in data]
    data = [text.replace('[SNT] ', '').strip() if text.startswith('[SNT] ') else text for text in data]
    data = [text.strip('[').strip(']').strip() for text in data]

    for j, text in enumerate(data):
        text = text.replace('TRIPLE]', '[TRIPLE]')
        if '[TRIPLE]' in text:
            texts = text.split('[TRIPLE]')[-1].strip()
            words = texts.split()
            textss = ' '.join(words[3:]).strip('[').strip(']').strip()
            #print(j+1, textss)
            data[j] = textss
            if len(textss.split()) <= 4:
                data[j] = text
    data = [text.split('[[TRIPLE]')[0].strip() for text in data]
    data = [text.split('[SNT] [[TRIPLE]')[0].strip() for text in data]
    data = [text.split('[/SNT] [[TRIPLE]')[0].strip() for text in data]
    for key, value in replc.items():
        data = [text.replace(key, value).strip(']').strip() for text in data]
    #data = [text.replace('[SNT]', '').replace('[', '').strip() for text in data]
    return data

def process_co_ord(data):
    replc = {'[TRIPLE]': '', '[/TRIPLE]': '', 'TRIPLE':'', '"':'', '[':'', ']':'' }
    for key, value in replc.items():
        data = [text.replace(key, value).strip() for text in data]
    data = [re.sub(r'\s+', ' ', text).strip() for text in data]
    return data


def process_co_str(data):
    replc = {'[TRIPLE]': '', '[/TRIPLE]': '', 'TRIPLE':'', '"':'' }
    for key, value in replc.items():
        data = [text.replace(key, value).strip() for text in data]
    data = [re.sub(r'\d.\s', '', text).strip() for text in data]
    data = [re.sub(r'\s+', ' ', text).strip() for text in data]
    return data

def process_co_lex(data):
    replc = { '"':'', '>':'', '`':'', '   ':'. ', '...':'.', '..':'.' } #'[TRIPLE]': '', '[/TRIPLE]': '', 'TRIPLE':'',
    data = [text.split(':  ')[0].strip() for text in data]
    for key, value in replc.items():
        data = [text.replace(key, value).strip() for text in data]
    data = ['.'.join(text.split('.')[:-1])+'.' if '?' in text.split('.')[-1] else text for text in data]
    data = [re.sub(r'\d.\s', '', text).strip() for text in data]
    data = [re.sub(r'\s+', ' ', text).replace('..', '.').strip() for text in data]
    data = [text if text.strip().endswith('.') else text+'.' for text in data]
    return data

def process_co_reg(data):
    data = [text.replace(': _', ':_').split(': ')[-1].strip() for text in data]
    data = [text.split('[/TRIPLE] [/SNT] """')[1] if len(text.split('[/TRIPLE] [/SNT] """')) > 1 else text.split('[/TRIPLE] [/SNT] """')[0] for text in data]
    data = [text.split('""" [SNT] [TRIPLE]')[0] for text in data]
    replc = {'[TRIPLE]': '', '[/TRIPLE]': '', 'TRIPLE':'', '"':'', '[SNT]':'', '[/SNT]':'' }
    for key, value in replc.items():
        data = [text.replace(key, value).strip() for text in data]
    # data = [re.sub(r'\d.\s', '', text).strip() for text in data]
    data = [re.sub(r'\s+', ' ', text).replace('...', '.').replace('..', '.').strip() for text in data]
    return data


task = 'reg' #'end2end'
model = 'cohere' #'mistral7b_struct'
work_folder_path = f'/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/{task}/rawfiles/{model}'
input_folder_path = f'/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/{task}/{model}'
files = os.listdir(input_folder_path)

for file in files:
    # Rename the file
    #file_original = file.split('.')[0]+ '_original.txt' if file.endswith('.txt') else file
    #os.rename(os.path.join(input_folder_path, file), os.path.join(input_folder_path, file_original))

    # Read and process the data
    with open(os.path.join(work_folder_path, file), 'r') as f:
        data = f.read().split('\n')
        if task == 'ordering':
            cleaned_data = process_co_ord(data)
        elif task == 'structuring':
            #cleaned_data = process_str_data(data)
            cleaned_data = process_co_str(data)
        elif task == 'lexicalization':
            #cleaned_data = process_lex_data(data)
            cleaned_data = process_co_lex(data)
        elif task == 'reg':
            cleaned_data = process_co_reg(data)
        elif task == 'end2end':
            #cleaned_data = process_e2e_data(data) #mistral7b_e2e
            cleaned_data = process_mis_str(data) #mistral7b_struct
        else:
            print('Put in a valid task')
        


    # Save the cleaned data into the original file name
    with open(os.path.join(input_folder_path, file), 'w') as f:
        f.write('\n'.join(cleaned_data))
