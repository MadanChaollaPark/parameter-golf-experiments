"""Bundle model, training and frozen evaluation into one counted train_gpt.py."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=(root/'train_gpt.py').read_text()
base=base.replace('from __future__ import annotations','from __future__ import annotations\nimport time\nSCRIPT_START=time.perf_counter()',1)
base=base.replace('def main() -> None:','def train_main() -> None:',1)
base=base.rsplit('\nif __name__ == "__main__":',1)[0]
win=(root/'eval_windows.py').read_text().split("\nif __name__=='__main__':",1)[0]
eval_code=(root/'evaluate.py').read_text();eval_code=eval_code[eval_code.index('import argparse'):]
eval_code=eval_code.replace('import train_gpt as m','m=sys.modules[__name__]').replace('from eval_windows import windows,prove_coverage\n','').replace('def main():','def evaluation_main():',1)
eval_code=eval_code.split("\nif __name__=='__main__':",1)[0]
output=base+'\n\n# Frozen evaluation: all required code is counted below.\n'+win+'\n'+eval_code+'''

if __name__ == '__main__':
    if '--evaluate' in sys.argv:
        sys.argv.remove('--evaluate')
        evaluation_main()
    else:
        train_main()
'''
(root/'submission').mkdir(exist_ok=True)
(root/'submission/train_gpt.py').write_text(output)
print('Bundled counted code bytes:',len(output.encode()))
