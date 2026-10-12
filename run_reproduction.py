"""Portable verification of retained forecasts. Does not train any model."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parent
def unpack(path,dest):
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for item in z.infolist():
            target=(dest/item.filename).resolve()
            if not target.is_relative_to(dest.resolve()) or item.is_dir():raise ValueError('Invalid archive member')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figures',action='store_true');args=ap.parse_args()
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        h,rel=line.split('  ',1)
        optional=rel.startswith('optional/') or rel=='requirements-figures.txt'
        if optional and not args.figures and not (ROOT/rel).exists():continue
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    unpack(ROOT/'data/frozen_core.zip',ROOT/'results/inputs')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',MPLCONFIGDIR=str(ROOT/'results/mplconfig'))
    for name in ['evaluate_core.py','evaluate_flow.py','evaluate_s5.py','evaluate_s9.py','evaluate_ranges.py']:
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'code'/name)],cwd=ROOT,env=env,check=True)
    s9=json.loads((ROOT/'results/s9_recovery_summary.json').read_text())
    expected={'Soy Z1 endpoint identity '+x for x in ['A010','A050','A090','A110','A150','A190']}
    assert {x['check'] for x in s9['failed']}==expected
    assert all(x['tolerance']==1e-10 for x in s9['failed'])
    if args.figures:
        unpack(ROOT/'optional/figure_inputs.zip',ROOT/'results/figure_inputs')
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'optional/render_figures.py')],cwd=ROOT,env=env,check=True)
    print('Frozen-result reproduction completed. Six documented strict S9 precision failures remain. Level B: NOT_RUN.')
if __name__=='__main__':main()
