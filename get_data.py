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

def save_to_text_file(file_list, output_file):
    with open(output_file, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)

def common_elements(list1, list2):
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

# 디렉토리 경로 설정
shake_directory_paths = ['/home/scar/Desktop/woojeong/final/datasets/training/source/bright_RL', '/home/scar/Desktop/woojeong/final/datasets/training/source/bright_UD', '/home/scar/Desktop/woojeong/final/datasets/training/source/dark_RL', '/home/scar/Desktop/woojeong/final/datasets/training/source/dark_UD']
focus_directory_paths = ['/home/scar/Desktop/woojeong/final/datasets/training/source/bright_FF', '/home/scar/Desktop/woojeong/final/datasets/training/source/bright_RF', '/home/scar/Desktop/woojeong/final/datasets/training/source/dark_FF', '/home/scar/Desktop/woojeong/final/datasets/training/source/dark_RF']
gt_directory_paths = ['/home/scar/Desktop/woojeong/final/datasets/training/bright_GT', '/home/scar/Desktop/woojeong/final/datasets/training/dark_GT']

directory_paths = shake_directory_paths+focus_directory_paths
output_file = 'datasets/all.txt'

image_files_list = []
gt_image_files_list = []

# 이미지 파일 리스트 가져오기
for directory_path in directory_paths:
    image_files_list.extend(get_image_files(directory_path))

for directory_path in gt_directory_paths:
    gt_image_files_list.extend(get_image_files(directory_path))


image_files_list = [os.path.basename(fn) for fn in image_files_list]
gt_image_files_list = [os.path.basename(fn) for fn in gt_image_files_list]

# 공통된 요소만 남기기
result = common_elements(image_files_list, gt_image_files_list)

# 이미지 파일 리스트를 텍스트 파일로 저장
save_to_text_file(result, output_file)

print("이미지 파일 리스트가 '%s' 파일에 저장되었습니다." % output_file)

