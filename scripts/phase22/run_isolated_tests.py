"""Run the exact registered suite in isolated CPU-only module processes.

Isolation releases cached arrays between modules while other user work occupies RAM.
Registration is read from tests/run_tests.py, so no registered module is omitted.
"""
import argparse,ast,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--tag',default='full_tests_isolated');args=ap.parse_args()
assert args.tag.replace('_','').isalnum()
output=ROOT/f'reports/phase22/{args.tag}.json';assert not output.exists()
tree=ast.parse((ROOT/'tests/run_tests.py').read_text())
aliases={n.asname:n.name for node in ast.walk(tree) if isinstance(node,ast.Import) for n in node.names}
mods=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mods' for t in node.targets))
modules=[aliases[n.id] for n in mods.elts]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
started=time.monotonic();results=[];passed=failed=0
r=subprocess.run([sys.executable,str(ROOT/'tests/test_stochastic_protocol.py')],cwd=ROOT,env=env,capture_output=True,text=True)
passed+=int(r.returncode==0);failed+=int(r.returncode!=0)
results.append({'module':'tests.test_stochastic_protocol','passed':int(r.returncode==0),'failed':int(r.returncode!=0),'stdout':r.stdout[-5000:],'stderr':r.stderr[-5000:]})
print('stochastic_protocol',r.returncode,flush=True)
child='''import importlib,json,sys,time,traceback
m=importlib.import_module(sys.argv[1]); rows=[]
for n in sorted(dir(m)):
 if not n.startswith('test_'):continue
 start=time.monotonic()
 try:getattr(m,n)(); rows.append({'name':n,'pass':True,'seconds':time.monotonic()-start})
 except Exception:rows.append({'name':n,'pass':False,'traceback':traceback.format_exc()})
print('P22_TEST_RESULT='+json.dumps(rows),flush=True)
raise SystemExit(any(not r['pass'] for r in rows))
'''
for module in modules:
    r=subprocess.run([sys.executable,'-u','-c',child,module],cwd=ROOT,env=env,capture_output=True,text=True)
    tail=[line for line in r.stdout.splitlines() if line.startswith('P22_TEST_RESULT=')]
    rows=json.loads(tail[-1].split('=',1)[1]) if tail else []
    ok=sum(x['pass'] for x in rows);bad=sum(not x['pass'] for x in rows)
    if not tail or (r.returncode and not bad):bad+=1
    passed+=ok;failed+=bad
    results.append({'module':module,'passed':ok,'failed':bad,'tests':rows,'exit_code':r.returncode,'stderr':r.stderr[-5000:]})
    print(json.dumps({'module':module,'passed':ok,'failed':bad}),flush=True)
assert not output.exists()
output.write_text(json.dumps({'mode':'Exact tests/run_tests.py registration, module-isolated, CUDA hidden','passed':passed,'failed':failed,'registered_modules':modules,'seconds':time.monotonic()-started,'results':results},indent=2)+'\n')
print(f'{passed} passed, {failed} failed',flush=True)
raise SystemExit(bool(failed))
