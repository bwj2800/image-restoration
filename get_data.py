import os

def get_image_files(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 이미지 파일의 확장자들
    image_files = []

    # 디렉토리 탐색
    for root, _, files in os.walk(directory):
        for file in files:
            # 파일이 이미지 파일인지 확인
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return image_files

def get_label_files(directory):
    label_files = []

    # 디렉토리 탐색
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.json'):
                label_files.append(os.path.join(root, file))

    return label_files

def save_to_text_file(file_list, output_file):
    with open(output_file, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)

def common_images(list1, list2):
    common_prefixes = set()

    for item1 in list1:
        for item2 in list2:
            suffixes = ['_FF.jpg', '_RF.jpg', '_UD.jpg', '_RL.jpg']
            for suffix in suffixes:
                prefix1 = item1.split(suffix)[0]
                prefix2 = item2.split('_GT.jpg')[0]
                
                if prefix1 == prefix2:
                    common_prefixes.add(item1)

    return list(common_prefixes)


def common_labels(list1, list2):
    common_prefixes = set()

    for item1 in list1:
        for item2 in list2:
            suffixes = ['_FF.json', '_RF.json', '_UD.json', '_RL.json']
            for suffix in suffixes:
                prefix1 = item1.split(suffix)[0]
                prefix2 = item2.split('_GT.json')[0]
                
                if prefix1 == prefix2:
                    common_prefixes.add(item1)

    return list(common_prefixes)


def common_elements(list1, list2):
    common_prefixes = set()

    for item1 in list1:
        for item2 in list2:
            prefix1 = item1.split('.jpg')[0]
            prefix2 = item2.split('.json')[0]
                
            if prefix1 == prefix2:
                common_prefixes.add(item1)

    return list(common_prefixes)

# 디렉토리 경로 설정
image_path='/home/scar/Desktop/woojeong/final/datasets/training'
val_image_path='/home/scar/Desktop/woojeong/final/datasets/validation'
shake_directory_paths = [val_image_path+'/source/bright_RL', val_image_path+'/source/bright_UD', val_image_path+'/source/dark_RL', val_image_path+'/source/dark_UD']
focus_directory_paths = [val_image_path+'/source/bright_FF', val_image_path+'/source/bright_RF', val_image_path+'/source/dark_FF', val_image_path+'/source/dark_RF']
gt_directory_paths = [val_image_path+'/bright_GT', val_image_path+'/dark_GT']

label_path='/home/scar/Desktop/woojeong/final/datasets/training/label'
val_label_path='/home/scar/Desktop/woojeong/final/datasets/validation/label'
shake_label_directory_paths = [val_label_path+'/bright_RL', val_label_path+'/dark_RL', val_label_path+'/bright_UD', val_label_path+'/dark_UD']
focus_label_directory_paths = [val_label_path+'/bright_RF', val_label_path+'/dark_RF', val_label_path+'/bright_FF', val_label_path+'/dark_FF']
gt_label_directory_paths = [val_label_path+'/bright_GT', val_label_path+'/dark_GT']


directory_paths = shake_directory_paths+focus_directory_paths
label_directory_paths = shake_label_directory_paths+focus_label_directory_paths
output_file = 'datasets/new_val_all.txt'

image_files_list = []
gt_image_files_list = []
label_files_list = []
gt_label_files_list = []

# 이미지 파일 리스트 가져오기
for directory_path in directory_paths:
    image_files_list.extend(get_image_files(directory_path))

for directory_path in gt_directory_paths:
    gt_image_files_list.extend(get_image_files(directory_path))

# label 파일 리스트 가져오기
for directory_path in label_directory_paths:
    label_files_list.extend(get_label_files(directory_path))

for directory_path in gt_label_directory_paths:
    gt_label_files_list.extend(get_label_files(directory_path))


image_files_list = [os.path.basename(fn) for fn in image_files_list]
gt_image_files_list = [os.path.basename(fn) for fn in gt_image_files_list]
label_files_list = [os.path.basename(fn) for fn in label_files_list]
gt_label_files_list = [os.path.basename(fn) for fn in gt_label_files_list]

# 공통된 요소만 남기기
img_result = common_images(image_files_list, gt_image_files_list)
print(len(img_result))
label_result = common_labels(label_files_list, gt_label_files_list)
print(len(label_result))
result = common_elements(img_result, label_result)
print(len(result))

# 이미지 파일 리스트를 텍스트 파일로 저장
save_to_text_file(result, output_file)

print("이미지 파일 리스트가 '%s' 파일에 저장되었습니다." % output_file)

