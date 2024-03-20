# code for checking json files (label) 
# since some data have "Learning_Data_Info" column instead of "Learning_Data_Info."

import re
import json

root="datasets/validation"
with open("datasets/new_val_focus.txt", 'r') as f:
    files = [x.strip() for x in f.readlines()]

for f in files:
    file_name=re.split(r'_|\.', f)
    label_path=root+'/label'
    gt_label_path=root+'/label'
    if file_name[2].startswith('OH') or file_name[2].startswith('IH'): #밝은 조도
        label_path+='/bright_'
        gt_label_path+='/bright_GT/'
    else:
        label_path+='/dark_'
        gt_label_path+='/dark_GT/'

    label_path+=file_name[-2]+'/'

    print(gt_label_path+f[:-6]+'GT.json')
    with open(gt_label_path+f[:-6]+'GT.json', 'r', encoding='utf-8-sig') as json_file:
        gt_data = json.load(json_file)
    type_value = gt_data["Learning_Data_Info."]["Annotations"][0]["Type_value"]

    print(label_path+f[:-3]+'json')
    with open(label_path+f[:-3]+'json', 'r', encoding='utf-8-sig') as json_file:
        data = json.load(json_file)
    type_value = data["Learning_Data_Info."]["Annotations"][0]["Type_value"]

