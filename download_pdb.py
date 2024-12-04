def extract_descriptions(fasta_file):
    descriptions = []
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Remove '>' and strip whitespace
                desc = line[1:].strip()
                # Only keep descriptions that don't start with 'T'
                if not desc.startswith('T'):
                    descriptions.append(desc)
    return descriptions

def parse_pdb_chains(descriptions):
    pdb_chains = []
    for desc in descriptions:
        # Split on underscore to separate PDB ID and chain
        if '_' in desc:
            pdb_id, chain = desc.split('_')
            pdb_chains.append((pdb_id.lower(), chain))
    return pdb_chains

# Example usage
fasta_file = "query_sequences.fasta"
descriptions = extract_descriptions(fasta_file)
pdb_chains = parse_pdb_chains(descriptions)

print("Extracted descriptions:", descriptions)
print("PDB chains:", pdb_chains)