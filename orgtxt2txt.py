import os

def process_las_files(folder_path):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(folder_path):
        # 筛选出不包含'out'且以'.txt'结尾的文件
        if 'out' not in filename and filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # 读取原始文件
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 处理每一行，只保留第1、2、3和7列，并附加一列为0
            processed_lines = []
            for line in lines:
                columns = line.split()
                if len(columns) >= 7:
                    new_line = ' '.join([columns[0], columns[1], columns[2], columns[6], '0']) + '\n'
                    processed_lines.append(new_line)

            # 获取文件名中的编号
            file_num = filename.split('_')[1].split('.')[0]
            new_filename = f'scene_{file_num}.txt'
            new_file_path = os.path.join(folder_path, new_filename)

            # 将处理后的内容写入新文件
            with open(new_file_path, 'w') as new_file:
                new_file.writelines(processed_lines)


# 指定文件夹路径
folder_path = r'/mnt/d/Area_22'
process_las_files(folder_path)