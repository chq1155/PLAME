import csv, os
from TMscore import TMscore

def process_pdb_pairs(folder1, folder2, output_csv):
    # 获取两个文件夹中的所有 .pdb 文件
    files1 = {f for f in os.listdir(folder1) if f.endswith('.pdb')}
    files2 = {f for f in os.listdir(folder2) if f.endswith('.pdb')}

    # 找到共同的文件名
    common_files = files1.intersection(files2)

    results = []

    for file in common_files:
        path1 = os.path.join(folder1, file)
        path2 = os.path.join(folder2, file)

        # 调用 TMscore 函数
        lengths, tmscore_results = TMscore(path1, path2, seq=True)

        # 提取 TMscore 和 RMSD
        tmscore = tmscore_results[0]
        rmsd = tmscore_results[1]

        print(f"File: {file}, TMscore: {tmscore}, RMSD: {rmsd}")

        # 将结果添加到列表中
        results.append([file, tmscore, rmsd])

    # 将结果写入 CSV 文件
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'TMscore', 'RMSD'])  # 写入表头
        writer.writerows(results)

    print(f"Results have been saved to {output_csv}")

folder1 = "./casp14/ESMFold-pred"
folder2 = "./casp14/gt"
output_csv = "tmscore_results.csv"

process_pdb_pairs(folder1, folder2, output_csv)