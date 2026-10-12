from pathlib import Path
import csv,hashlib,json,re,sys
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE/'results/inputs'
TEST=PACKAGE/'results'; TEST.mkdir(exist_ok=True)
AUDIT=TEST
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj):(TEST/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def outcsv(path,rows,fields=None):
    rows=list(rows)
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def text(p):return Path(p).read_text(encoding='utf-8-sig')
