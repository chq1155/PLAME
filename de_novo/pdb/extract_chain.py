from Bio import PDB
import os

def extract_chains_from_multiple_pdbs(pdb_chain_pairs, output_dir=None):
    """
    从多个PDB文件中提取指定的链结构
    
    参数:
        pdb_chain_pairs: 字典，格式为 {pdb文件路径: [链ID列表]}
                        例如 {'1abc.pdb': ['A', 'B'], '2xyz.pdb': ['C']}
        output_dir: 输出目录，默认为None（保存在当前目录）
    
    返回:
        dict: 包含成功/失败结果的字典
    """
    results = {}
    parser = PDB.PDBParser(QUIET=True)
    
    # 如果指定了输出目录，确保它存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for pdb_file, chain_ids in pdb_chain_pairs.items():
        # 获取PDB ID（文件名的基本部分）
        pdb_id = os.path.basename(pdb_file).split('.')[0]
        results[pdb_id] = {'success': [], 'failed': []}
        
        try:
            # 解析PDB文件
            structure = parser.get_structure(pdb_id, pdb_file)
            
            # 处理每个需要提取的链
            for chain_id in chain_ids:
                try:
                    # 创建新结构只包含指定链
                    new_structure = PDB.Structure.Structure(f"{pdb_id}_{chain_id}")
                    model = PDB.Model.Model(0)
                    new_structure.add(model)
                    
                    # 查找并添加指定链
                    chain_found = False
                    for chain in structure[0]:
                        if chain.id == chain_id:
                            model.add(chain)
                            chain_found = True
                            break
                    
                    if not chain_found:
                        print(f"在文件 {pdb_file} 中未找到链ID: {chain_id}")
                        results[pdb_id]['failed'].append(chain_id)
                        continue
                    
                    # 确定输出文件路径
                    if output_dir:
                        output_file = os.path.join(output_dir, f"{pdb_id}_chain_{chain_id}.pdb")
                    else:
                        output_file = f"{pdb_id}_chain_{chain_id}.pdb"
                    
                    # 写入输出文件
                    io = PDB.PDBIO()
                    io.set_structure(new_structure)
                    io.save(output_file)
                    
                    print(f"成功从 {pdb_file} 提取链 {chain_id} 并保存到 {output_file}")
                    results[pdb_id]['success'].append(chain_id)
                    
                except Exception as e:
                    print(f"处理文件 {pdb_file} 的链 {chain_id} 时出错: {str(e)}")
                    results[pdb_id]['failed'].append(chain_id)
        
        except Exception as e:
            print(f"解析文件 {pdb_file} 时出错: {str(e)}")
            results[pdb_id]['failed'].extend(chain_ids)
    
    return results

# 使用示例
if __name__ == "__main__":
    # 定义需要处理的PDB文件和对应的链ID
    pdb_chains = {
        "8cyk.pdb": ["A"],  # 从example1.pdb提取A链和B链
        "8sk7.pdb": ["C"],       # 从example2.pdb提取C链
        "8tnm.pdb": ["A"],   # 从example3.pdb提取A链和D链
        "8tno.pdb": ["A"],
    }
    
    # 执行提取，并将结果保存到"extracted_chains"目录
    results = extract_chains_from_multiple_pdbs(pdb_chains, "../chains")
    
    # 打印结果摘要
    print("\n提取结果摘要:")
    for pdb_id, result in results.items():
        print(f"{pdb_id}:")
        print(f"  成功: {', '.join(result['success'])}" if result['success'] else "  成功: 无")
        print(f"  失败: {', '.join(result['failed'])}" if result['failed'] else "  失败: 无")