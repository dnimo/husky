input_file_path = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/data/data_for_tain_tokenizer.txt'
output_file_path = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/data/del_none_data_for_train_tokenizer.txt'

# 打开输入文件
with open(input_file_path, 'r') as infile:
    # 读取文件内容
    content = infile.read()

# 去除空行
content_without_empty_lines = "\n".join([line for line in content.splitlines() if line.strip()])

# 打开输出文件并写入处理后的内容
with open(output_file_path, 'w') as outfile:
    outfile.write(content_without_empty_lines)

print("空行已被移除，并保存到output.txt文件中。")
