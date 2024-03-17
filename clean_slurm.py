import re

with open("slurm-203997.out", 'r') as file:
    file_content = file.read()

# Pattern for matching the desired text
pattern = r"INFO:root:Polling for Model: Final status of polling for model_process: {'completed': True,(.*?)INFO:root:Single Poll for Model: Status of polling for model_process: {'completed': True, 'data': "

# Find all matches of the pattern in the file content
matches = re.findall(pattern, file_content, re.DOTALL)

# Initialize an empty list to store the extracted data
extracted_data = []

# Iterate over the matches
for match in matches:
    # Split the match by the specified delimiter
    parts = match.split(",")
    # Get the first part, remove the leading and trailing spaces and single quotes, and append it to the extracted data list
    extracted_data.append(parts[0].strip().strip("'"))

# Print the extracted data
for data in extracted_data[:644]:
    print(data)

# with open("slurm-203997.out", 'r') as file:
#     file = file.read()

# # text = file
# # text = file.split('INFO:root:Polling for Model: Start polling for model_process')
# text = file.split("INFO:root:Polling for Model: Final status of polling for model_process: {'completed': True,")
# g = text[0].split("INFO:root:Single Poll for Model: Status of polling for model_process: {'completed': True, 'data': ")
# print(g[1].split(',')[0].replace("'", ''))

# data_match = re.search(r"'data': '([^']+)'", file)
# if data_match:
#     data_text = data_match.group(1)
#     print(data_text)

# for i, final in enumerate(text):
#     final = final.split("INFO:root:Single Poll for Model: Status of polling for model_process: {'completed': True, 'data': ")
#     for j, check in enumerate(final):

# for line in text:
#     if isinstance(line, str):
#         data_match = re.search(r"'data': '([^']+)'", line)
#         if data_match:
#             data_text = data_match.group(1)
#             # print(data_text)
#             # f = data_text[:644]
#             # print(f)
#             # extracts = data_text.split('\n') 
#             # print(extracts)
#     else:
#         print("The provided text is not a string.")