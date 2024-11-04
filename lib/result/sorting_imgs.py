import os
import shutil

def move_files_with_all_keywords(src_dir, dst_dir, keywords):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    all_files = os.listdir(src_dir)
    for file in all_files:
        src_file = os.path.join(src_dir, file)
        if os.path.isfile(src_file):
            if keywords[0] in file and keywords[1] in file:
                dst_file = os.path.join(dst_dir, file)
                shutil.move(src_file, dst_file)
                print(f'Moved: {src_file} -> {dst_file}')

if __name__ == '__main__':
    src_directory = '../save/img/all/'
    defenses = {'fltrust'}
    features = {'ma', 'ba', 'time', 'loss'}
    for defense in defenses:
        for feature in features:
            dst_directory = f'../save/img/{defense}_{feature}s'
            keywords = [defense, feature]
            move_files_with_all_keywords(src_directory, dst_directory, keywords)
