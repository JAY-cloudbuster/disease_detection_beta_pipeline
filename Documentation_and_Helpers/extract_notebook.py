import json
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else 'MediVLM_MIMIC_Sample_Train.ipynb'
output_file = input_file.replace('.ipynb', '_summary.txt')

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(output_file, 'w', encoding='utf-8') as out:
    for c in nb.get('cells', []):
        cell_type = c.get('cell_type', 'unknown')
        out.write(f'--- {cell_type} ---\n')
        source = c.get('source', [])
        if isinstance(source, list):
            out.write(''.join(source) + '\n\n')
        else:
            out.write(source + '\n\n')
