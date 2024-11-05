import os
import re
import shutil
from datetime import datetime

def fix_and_check_a3m_format(file_path):
    errors = []
    fixed_lines = []
    needs_fix = False
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查行中间是否有'>'
        if '>' in line and not line.startswith('>'):
            needs_fix = True
            # 在'>'处分割行
            parts = line.split('>')
            # 处理第一部分
            fixed_lines.append(parts[0].strip() + '\n')
            # 处理剩余部分
            for part in parts[1:]:
                if part.strip():  # 确保不是空字符串
                    fixed_lines.append('>' + part.strip() + '\n')
            errors.append(f"Line {i+1}: Fixed merged description line")
        else:
            fixed_lines.append(line)
        
        i += 1
    
    if needs_fix:
        # 创建备份文件
        backup_path = file_path + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy2(file_path, backup_path)
        
        # 写入修正后的内容
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)
    
    return errors, needs_fix

def process_folder(folder_path):
    print(f"Processing a3m files in folder: {folder_path}")
    print("-" * 50)
    
    fixed_files = []
    error_files = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.a3m'):
            file_path = os.path.join(folder_path, filename)
            errors, was_fixed = fix_and_check_a3m_format(file_path)
            
            if errors:
                error_files.append(filename)
                print(f"\nProcessing file: {filename}")
                for error in errors:
                    print(error)
                if was_fixed:
                    fixed_files.append(filename)
                    print("File has been fixed and backup created.")
    
    print("\n" + "=" * 50)
    print("Processing Summary:")
    if fixed_files:
        print(f"\nFixed files ({len(fixed_files)}):")
        for file in fixed_files:
            print(f"- {file}")
    if not error_files:
        print("\nNo format errors found in any a3m files.")
    print("\nBackup files have been created for all modified files.")

if __name__ == "__main__":
    folder_path = input("请输入要处理的文件夹路径: ")
    process_folder(folder_path)