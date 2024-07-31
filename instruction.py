
data_examples = {
    1: {
        'Input_ordering': "[TRIPLE] Asam_pedas country Malaysia [/TRIPLE] [TRIPLE] Malaysia ethnicGroup Malaysian_Chinese [/TRIPLE] [TRIPLE] Malaysia ethnicGroup Malaysian_Malay [/TRIPLE] [TRIPLE] Asam_pedas region Sumatra [/TRIPLE]",
        'Output_ordering': "country ethnicGroup ethnicGroup region",
        'Input_structuring': "[TRIPLE] Elliot_See almaMater University_of_Texas_at_Austin [/TRIPLE] [TRIPLE] University_of_Texas_at_Austin affiliations University_of_Texas_System [/TRIPLE] [TRIPLE] University_of_Texas_at_Austin compete_in Big_12_Conference [/TRIPLE] [TRIPLE] University_of_Texas_at_Austin mascot Hook_em_(mascot) [/TRIPLE] [TRIPLE] Elliot_See deathPlace St._Louis [/TRIPLE] [TRIPLE] St._Louis leaderName Francis_G._Slay [/TRIPLE]",
        'Output_structuring': "[SNT] almaMater [/SNT] [SNT] affiliations compete_in [/SNT] [SNT] mascot [/SNT] [SNT] deathPlace leaderName [/SNT]",
        'Input_lexicalization': "[SNT] [TRIPLE] Athens_International_Airport runwayLength 4000.0 [/TRIPLE] [/SNT]",
        'Output_lexicalization': "ENTITY-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a runway length of ENTITY-2.",
        'Input_reg': "A.S. Livorno Calcio vp[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] manage by . vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be vp[aspect=simple,tense=past,voice=active,person=null,number=null] attach to Real Madrid C.F. . Christian Panucci vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be currently vp[aspect=simple,tense=past,voice=active,person=null,number=null] attach to Genoa C.F.C.. [Christian Panucci]", 
        'Output_reg': "Christian Panucci",
        'Input_end2end': '[TRIPLE] A_Loyal_Character_Dancer ISBN_number "1-56947-301-3" [/TRIPLE] [TRIPLE] A_Loyal_Character_Dancer OCLC_number 49805501 [/TRIPLE] [TRIPLE] A_Loyal_Character_Dancer author Qiu_Xiaolong [/TRIPLE] [TRIPLE] A_Loyal_Character_Dancer mediaType "Print" [/TRIPLE]',
        'Output_end2end': 'The book, A Loyal Character Dancer, has the ISBN number of 1-56947-301-3 and The OCLC number is 49805501. It was penned by Qiu Xiaolong and is in print.',
    },
    2: {
        'Input_ordering': '[TRIPLE] Bananaman broadcastedBy BBC [/TRIPLE] [TRIPLE] Bananaman creator Steve_Bright [/TRIPLE] [TRIPLE] Bananaman firstAired "1983-10-03" [/TRIPLE] [TRIPLE] Bananaman lastAired "1986-04-15" [/TRIPLE] [TRIPLE] Bananaman starring Graeme_Garden [/TRIPLE]',
        'Output_ordering': "broadcastedBy firstAired lastAired creator starring",
        'Input_structuring': "[TRIPLE] Dessert dishVariation Sandesh_(confectionery) [/TRIPLE] [TRIPLE] Baked_Alaska course Dessert [/TRIPLE] [TRIPLE] Baked_Alaska country France [/TRIPLE] [TRIPLE] France language French_language [/TRIPLE]",
        'Output_structuring': "[SNT] dishVariation course [/SNT] [SNT] country language [/SNT]",
        'Input_lexicalization': "[SNT] [TRIPLE] A_Wizard_of_Mars country United_States [/TRIPLE] [TRIPLE] United_States language English_language [/TRIPLE] [TRIPLE] English_language spokenIn Great_Britain [/TRIPLE] [/SNT] [SNT] [TRIPLE] United_States leaderName Barack_Obama [/TRIPLE] [TRIPLE] United_States ethnicGroup Asian_Americans [/TRIPLE] [/SNT]",
        'Output_lexicalization': "ENTITY-1 VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] publish in ENTITY-2 where ENTITY-3 ( ENTITY-3 of ENTITY-4 ) VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] speak . ENTITY-5 VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be ENTITY-5 of ENTITY-2 and ENTITY-6 VP[aspect=simple,tense=present,voice=active,person=non-3rd,number=plural] be amongst DT[form=defined] the population there.",
        'Input_reg': "dt[form=defined] the leader of vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be Klaus Iohannis . vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be home to different ethnic groups , one of which vp[aspect=simple,tense=present,voice=active,person=non-3rd,number=plural] be Germans of Romania . dt[form=defined] the capital city vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be Bucharest and Romania vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be dt[form=defined] the location of 1 Decembrie 1918 University. [Romania]", 
        'Output_reg': "The country",
        'Input_end2end': '[TRIPLE] 250_Delaware_Avenue cost "110_million_(dollars)" [/TRIPLE] [TRIPLE] 250_Delaware_Avenue floorArea 30843.8_(square_metres) [/TRIPLE] [TRIPLE] 250_Delaware_Avenue floorCount 12 [/TRIPLE] [TRIPLE] 250_Delaware_Avenue location United_States [/TRIPLE]',
        'Output_end2end': '250 Delaware Avenue was built for 110 million dollars and is located in the United States. It has 12 floors with a total area of 30843.8 square metres.',
    },
    3: {
        'Input_ordering': '[TRIPLE] AWH_Engineering_College academicStaffSize 250 [/TRIPLE] [TRIPLE] AWH_Engineering_College established 2001 [/TRIPLE] [TRIPLE] AWH_Engineering_College state Kerala [/TRIPLE]',
        'Output_ordering': "academicStaffSize state established",
        'Input_structuring': '[TRIPLE] Turkey largestCity Istanbul [/TRIPLE] [TRIPLE] Turkey capital Ankara [/TRIPLE] [TRIPLE] Turkey leader Ahmet_Davutoğlu [/TRIPLE] [TRIPLE] Atatürk_Monument_(İzmir) location Turkey [/TRIPLE] [TRIPLE] Atatürk_Monument_(İzmir) material "Bronze" [/TRIPLE] [TRIPLE] Atatürk_Monument_(İzmir) designer Pietro_Canonica [/TRIPLE] [TRIPLE] Atatürk_Monument_(İzmir) inaugurationDate "1932-07-27" [/TRIPLE]',
        'Output_structuring': "[SNT] largestCity capital [/SNT] [SNT] leader location material designer inaugurationDate [/SNT]",
        'Input_lexicalization': "[SNT] [TRIPLE] Karnataka has_to_its_northeast Telangana [/TRIPLE] [TRIPLE] Karnataka has_to_its_west Arabian_Sea [/TRIPLE] [/SNT] [SNT] [TRIPLE] Acharya_Institute_of_Technology state Karnataka [/TRIPLE] [TRIPLE] Acharya_Institute_of_Technology city Bangalore [/TRIPLE] [/SNT] [SNT] [TRIPLE] Acharya_Institute_of_Technology sportsOffered Tennis [/TRIPLE] [TRIPLE] Tennis sportsGoverningBody International_Tennis_Federation [/TRIPLE] [/SNT] [SNT] [TRIPLE] Acharya_Institute_of_Technology affiliation Visvesvaraya_Technological_University [/TRIPLE] [/SNT]",
        'Output_lexicalization': "DT[form=defined] the state of ENTITY-1 VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] position with ENTITY-2 to ENTITY-1 northeast and ENTITY-3 to ENTITY-1 west . DT[form=defined] the state VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be home to ENTITY-4 in DT[form=defined] the city of ENTITY-5 . ENTITY-4 VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] know for ENTITY-6 VP[aspect=progressive,tense=present,voice=active,person=null,number=null] be one of DT[form=defined] the sports VP[aspect=simple,tense=past,voice=active,person=null,number=null] offer which VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] govern by ENTITY-7 . ENTITY-4 also VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have strong connections with ENTITY-8.",
        'Input_reg': "Peter Stöger vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be dt[form=defined] the manager of 1. FC Köln and vp[aspect=simple,tense=present,voice=active,person=3rd,number=null] play for FC Admira Wacker Mödling . Peter Stöger vp[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] affiliate with SC Wiener Neustadt club and previously vp[aspect=simple,tense=past,voice=active,person=null,number=null] play for. [LASK Linz]", 
        'Output_reg': "The club LASK Linz",
        'Input_end2end': '[TRIPLE] Asam_pedas country Malaysia [/TRIPLE] [TRIPLE] Malaysia ethnicGroup Malaysian_Chinese [/TRIPLE] [TRIPLE] Malaysia ethnicGroup Malaysian_Malay [/TRIPLE] [TRIPLE] Asam_pedas region Malay_Peninsula [/TRIPLE]',
        'Output_end2end': 'Asam pedas is found in the Malay Peninsula and throughout Malaysia. The Malaysian Chinese and the Malaysian Malay are both ethnic groups found in the country.',
    },
    4: {
        'Input_ordering': '[TRIPLE] Massimo_Drago club S.S.D._Potenza_Calcio [/TRIPLE] [TRIPLE] A.C._Cesena manager Massimo_Drago [/TRIPLE]',
        'Output_ordering': "manager club",
        'Input_structuring': '[TRIPLE] Azerbaijan  leader  Artur_Rasizade [/TRIPLE], [TRIPLE] Baku_Turkish_Martyrs\'_Memorial  material  "Red granite and white marble" [/TRIPLE], [TRIPLE] Azerbaijan  leaderTitle  Prime_Minister_of_Azerbaijan [/TRIPLE], [TRIPLE] Baku_Turkish_Martyrs\'_Memorial  dedicatedTo  "Ottoman Army soldiers killed in the Battle of Baku" [/TRIPLE], [TRIPLE] Baku_Turkish_Martyrs\'_Memorial  location  Azerbaijan [/TRIPLE], [TRIPLE] Baku_Turkish_Martyrs\'_Memorial  nativeName  "Türk Şehitleri Anıtı" [/TRIPLE], [TRIPLE] Baku_Turkish_Martyrs\'_Memorial  designer  "Hüseyin Bütüner and Hilmi Güner" [/TRIPLE]',
        'Output_structuring': "[SNT] leader leaderTitle [/SNT] [SNT] location dedicatedTo [/SNT] [SNT] nativeName [/SNT] [SNT] designer material [/SNT]",
        'Input_lexicalization': "[SNT] [TRIPLE] Denmark leaderName Lars_Løkke_Rasmussen [/TRIPLE] [TRIPLE] School_of_Business_and_Social_Sciences_at_the_Aarhus_University country Denmark [/TRIPLE] [TRIPLE] School_of_Business_and_Social_Sciences_at_the_Aarhus_University city Aarhus [/TRIPLE] [/SNT] [SNT] [TRIPLE] School_of_Business_and_Social_Sciences_at_the_Aarhus_University established 1928 [/TRIPLE] [TRIPLE] School_of_Business_and_Social_Sciences_at_the_Aarhus_University affiliation European_University_Association [/TRIPLE] [/SNT] [SNT] [TRIPLE] School_of_Business_and_Social_Sciences_at_the_Aarhus_University numberOfStudents 16000 [/TRIPLE] [TRIPLE] School_of_Business_and_Social_Sciences_at_the_Aarhus_University academicStaffSize 737 [/TRIPLE] [/SNT]",
        'Output_lexicalization': "ENTITY-2 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the leader of ENTITY-1 where ENTITY-3 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in DT[form=defined] the city of ENTITY-4 . ENTITY-3 VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] establish in ENTITY-5 and VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] affiliate with ENTITY-6 . There VP[aspect=simple,tense=present,voice=active,person=non-3rd,number=plural] be ENTITY-7 students and ENTITY-8 academic staff.",
        'Input_reg': "vp[aspect=simple,tense=past,voice=passive,person=null,number=singular] bear in Glen Ridge, New Jersey on  1930-01-20  . real name vp[aspect=simple,tense=past,voice=active,person=null,number=singular] be  Edwin E. Aldrin, Jr.  . vp[aspect=simple,tense=past,voice=active,person=null,number=null] graduate from  Massachusetts Institute of Technology, Sc.D. 1963  . vp[aspect=simple,tense=past,voice=active,person=null,number=singular] be Fighter pilot and crew member of Apollo 11. [Buzz Aldrin]", 
        'Output_reg': "He",
        'Input_end2end': '[TRIPLE] Azerbaijan_Premier_League champions Qarabağ_FK [/TRIPLE] [TRIPLE] AZAL_PFK ground AZAL_Arena [/TRIPLE] [TRIPLE] AZAL_PFK league Azerbaijan_Premier_League [/TRIPLE] [TRIPLE] AZAL_PFK numberOfMembers 3500 [/TRIPLE] [TRIPLE] AZAL_PFK season 2014–15_Azerbaijan_Premier_League [/TRIPLE]',
        'Output_end2end': 'AZAL Arena, which holds 3500 fans, is the ground of AZAL PFK who played in the Azerbaijan Premier League in 2014-15. Qarabag FK have been champions of this league.',
    },
    5: {
        'Input_ordering': "[TRIPLE] United_States leaderName John_Roberts [/TRIPLE] [TRIPLE] United_States leaderName Paul_Ryan [/TRIPLE] [TRIPLE] United_States leaderTitle President_of_the_United_States [/TRIPLE] [TRIPLE] 250_Delaware_Avenue location United_States [/TRIPLE]",
        'Output_ordering': "leaderTitle leaderName leaderName location",
        'Input_structuring': "[TRIPLE] Ampara_Hospital region Ampara_District [/TRIPLE] [TRIPLE] Ampara_Hospital state Eastern_Province,_Sri_Lanka [/TRIPLE] [TRIPLE] Ampara_Hospital country Sri_Lanka [/TRIPLE] [TRIPLE] Eastern_Province,_Sri_Lanka leaderName Austin_Fernando [/TRIPLE]",
        'Output_structuring': "[SNT] region state country leaderName [/SNT]",
        'Input_lexicalization': '[SNT] [TRIPLE] Apollo_8 operator NASA [/TRIPLE] [TRIPLE] William_Anders was_a_crew_member_of Apollo_8 [/TRIPLE] [TRIPLE] William_Anders served_as_Chief_of_the_Astronaut_Office_in 1976 [/TRIPLE] [TRIPLE] William_Anders timeInSpace "8820.0"(minutes) [/TRIPLE] [TRIPLE] Apollo_8 backup_pilot Buzz_Aldrin [/TRIPLE] [TRIPLE] Apollo_8 crewMembers Frank_Borman [/TRIPLE] [/SNT]',
        'Output_lexicalization': "ENTITY-1 VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] operate by ENTITY-2 and VP[aspect=simple,tense=past,voice=active,person=null,number=null] have DT[form=defined] the VP[aspect=progressive,tense=present,voice=active,person=null,number=null] follow crew : ENTITY-3 who VP[aspect=simple,tense=past,voice=active,person=null,number=null] serve as Chief of DT[form=defined] the Astronaut Office in ENTITY-4 and VP[aspect=simple,tense=past,voice=active,person=null,number=null] spend ENTITY-5 in space ; ENTITY-6 who VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=undefined] a backup pilot and ENTITY-7 who VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=undefined] a crew member.",
        'Input_reg': "campus vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in  In Soldevanahalli, Acharya Dr. Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore – 560090.  . vp[aspect=simple,tense=past,voice=passive,person=null,number=singular] establish in 2000 and motto vp[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be ``  Nurturing Excellence     . vp[aspect=simple,tense=present,voice=active,person=3rd,number=null] have 700 postgraduate students and Acharya Institute of Technology vp[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] affiliate to Visvesvaraya Technological University. [Acharya Institute of Technology]", 
        'Output_reg': "It",
        'Input_end2end': '[TRIPLE] Turkey leaderName Ahmet_Davutoğlu [/TRIPLE] [TRIPLE] Atatürk_Monument_(İzmir) location Turkey [/TRIPLE] [TRIPLE] Atatürk_Monument_(İzmir) material "Bronze" [/TRIPLE]',
        'Output_end2end': "The Atatürk Monument made of bronze is located in İzmir, Turkey, whose leader is Ahmet Davutoğlu.",
    }
}

data_examples_struct = {
    1: {
        "Input_struct2sr": "[SNT] [TRIPLE] Atatürk_Monument_(İzmir) material 'Bronze' [/TRIPLE] [TRIPLE] Atatürk_Monument_(İzmir) inaugurationDate '1932-07-27' [/TRIPLE] [/SNT] [SNT] [TRIPLE] Atatürk_Monument_(İzmir) location Turkey [/TRIPLE] [TRIPLE] Turkey capital Ankara [/TRIPLE] [TRIPLE] Turkey largestCity Istanbul [/TRIPLE] [/SNT] [SNT] [TRIPLE] Turkey leaderName Ahmet_Davutoğlu [/TRIPLE] [TRIPLE] Turkey currency Turkish_lira [/TRIPLE] [/SNT]",
        "Output_struct2sr": "The Atatürk Monument is a bronze monument inaugurated on 27th July, 1932, in Izmir. It is found in Turkey, a country which has Ankara as its capital and Istanbul as its largest city. The leader of Turkey is called Ahmet Davutoğlu, and the currency is the Turkish lira."
    },
    2: {
        "Input_struct2sr": "[SNT] [TRIPLE] Turkey capital Ankara [/TRIPLE] [TRIPLE] Turkey largestCity Istanbul [/TRIPLE] [/SNT] [SNT] [TRIPLE] Turkey leader Ahmet_Davutoğlu [/TRIPLE] [TRIPLE] Turkey currency Turkish_lira [/TRIPLE] [/SNT] [SNT] [TRIPLE] Atatürk_Monument_(İzmir) location Turkey [/TRIPLE] [/SNT]",
        "Output_struct2sr": "The capital of Turkey is Ankara, although the largest city is Istanbul. The leader of Turkey is Ahmet Davutoglu and the currency is known as the Turkish lira. The Ataturk monument is located within the country."
    },
    3: {
        "Input_struct2sr": "[SNT] [TRIPLE] Antwerp_International_Airport cityServed Antwerp [/TRIPLE] [TRIPLE] Antwerp country Belgium [/TRIPLE] [TRIPLE] Belgium leaderName Philippe_of_Belgium [/TRIPLE] [TRIPLE] Belgium language French_language [/TRIPLE] [/SNT]",
        "Output_struct2sr": "Antwerp is served by Antwerp International Airport and is a popular tourism destination in Belgium where the leader is Philippe of Belgium and the French language is spoken."
    },
    4: {
        "Input_struct2sr": "[SNT] [TRIPLE] AWH_Engineering_College state Kerala [/TRIPLE] [TRIPLE] AWH_Engineering_College country India [/TRIPLE] [TRIPLE] AWH_Engineering_College established 2001 [/TRIPLE] [/SNT] [SNT] [TRIPLE] India river Ganges [/TRIPLE] [TRIPLE] India largestCity Mumbai [/TRIPLE] [/SNT] [SNT] [TRIPLE] Kerala leaderName Kochi [/TRIPLE] [/SNT]",
        "Output_struct2sr": "The AWH Engineering College in Kerala, India was established in 2001. The Ganges is a river in India and its largest city is Mumbai. The leader of Kerala is Kochi."
    },
    5: {
        "Input_struct2sr": "[SNT] [TRIPLE] Atlanta country United_States [/TRIPLE] [TRIPLE] United_States capital Washington [/TRIPLE] [/SNT] [SNT] [TRIPLE] D.C. United_States ethnicGroup Asian_Americans [/TRIPLE] [/SNT]",
        "Output_struct2sr": "Atlanta is in the United States whose capital is Washington, D.C. Asian Americans are an ethnic group in the U.S."
    }
}


def generate_examples(data_examples, task_key):
    examples = ''
    for idx in range(1, 6):
        input_key = f"Input_{task_key}"
        output_key = f"Output_{task_key}"
        examples += f"\nInput: {data_examples[idx][input_key]} \nOutput: {data_examples[idx][output_key]} \n"
    return examples

def instruct_templates(model, source, task_key, pipeline=False):
    global data_examples, data_examples_struct, system_inst, instruction, instructions_struct

    examples = generate_examples(data_examples_struct if pipeline else data_examples, task_key)
    instruction = instruction_struct if pipeline else instruction
    if model in ['phi', 'cohere', 'gpt']:
        prompt = f'''{instruction}\nExamples:{examples}\nInput: {source}\nOutput:\n'''
    elif model in ['llama', 'mistral', 'vicuna']:
        prompt = f'''<s>[INST] <<SYS>> {system_inst}\n{instruction}\nExamples:{examples}<</SYS>>\nInput: {source}\nOutput:\n[/INST]'''
    else:
        ValueError('Select a model!')

    return prompt


system_inst = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being socially unbiased and safe. If you're unsure about an answer, it's okay to skip it, and please ensure not to provide incorrect information. Additionally, responses should be concise and informative."
instruction = f"You are provided with a text and your task it to generate texts from the last input provided following the examples given below. Do not include any information that cannot be directly inferred from the given text: "
#instruction="I would like you to generate a fluent and concise summaries or text in English based on the triples provided. Below you may find examples of the input triples and the expected summary outputs. Do not omit any triple information in the text or include any information that cannot be directly inferred from the given triples."
#instruction_struct="I would like you to generate a fluent and concise text in English based on the triples provided. Below you may find examples of the input triples and the expected textual outputs. Do not omit any triple information in the text or include any information that cannot be directly inferred from the given triples. Make sure to follow the format as found in the examples."
instruction_struct="Generate fluent and concise English text based on the provided triples. Refer to the examples below for input triples and their corresponding expected textual outputs. Ensure that all information from the triples is included in the generated text, following the sentence structuring indicated by the opening '[SNT]' and closing '[/SNT]' tags found in the input examples. Do not exclude any triple information or include any additional information not directly inferred from the given triples."
