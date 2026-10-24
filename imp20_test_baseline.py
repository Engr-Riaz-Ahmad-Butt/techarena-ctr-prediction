"""Isolated baseline-lock fixtures and exact one-parameter-change audit."""
import ast
import copy
import hashlib
import json
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
from config import ARTIFACT_DIR
FILES=['imp20_tree.py','imp10_leafwise.py']
HASHES={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
tree=ast.parse((ROOT/'imp20_tree.py').read_text())
loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_baseline')
guard=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit'))
assert isinstance(guard.body[0],ast.Assert)
guard.name='manifest_guard_only'
guard.body=[guard.body[0],ast.Return(value=ast.Constant(True))]
isolated=ast.fix_missing_locations(ast.Module(body=[loader,guard],type_ignores=[]))


def rejected(function,*args):
    try:function(*args)
    except AssertionError:return True
    return False


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='imp20_test_',dir=ARTIFACT_DIR) as directory:
        temp=Path(directory)
        assert temp.resolve().is_relative_to(ARTIFACT_DIR.resolve())
        payloads={6:b'isolated-day6-model-fixture',5:b'isolated-day5-model-fixture'}
        for day,payload in payloads.items():(temp/f'incumbent_day{day}.npz').write_bytes(payload)
        audit_path=temp/'accepted_audit.json'
        audit_path.write_text(json.dumps(dict(status='PASS',both_dates_improved=True)))
        manifest=dict(baseline='incumbent',sha256={str(day):hashlib.sha256(data).hexdigest() for day,data in payloads.items()},
                      audit_filename=audit_path.name,independent_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
        manifest_path=temp/'imp20_baseline.json'
        manifest_path.write_text(json.dumps(manifest))
        original_manifest=manifest_path.read_bytes()
        ns=dict(ROOT=temp,hashlib=hashlib,json=json)
        exec(compile(isolated,'imp20_tree.py','exec'),ns)
        frozen=ns['load_baseline']()
        assert frozen['_manifest_sha256']==hashlib.sha256(original_manifest).hexdigest()
        assert ns['manifest_guard_only'](6,frozen)
        mutated_days=[]
        for day,payload in payloads.items():
            path=temp/f'incumbent_day{day}.npz'
            path.write_bytes(payload+b'mutation')
            assert rejected(ns['load_baseline'])
            mutated_days.append(day)
            path.write_bytes(payload)
        original_audit=audit_path.read_bytes()
        audit_path.write_bytes(original_audit+b' ')
        assert rejected(ns['load_baseline'])
        audit_path.write_bytes(original_audit)
        # Hash-consistent but unsuccessful audits must still be rejected.
        for bad in [dict(status='FAIL',both_dates_improved=True),dict(status='PASS',both_dates_improved=False)]:
            audit_path.write_text(json.dumps(bad))
            altered=dict(manifest,independent_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
            manifest_path.write_text(json.dumps(altered))
            assert rejected(ns['load_baseline'])
        audit_path.write_bytes(original_audit)
        manifest_path.write_bytes(original_manifest)
        assert ns['manifest_guard_only'](6,frozen)
        manifest_path.write_bytes(original_manifest+b'\n')
        assert rejected(ns['manifest_guard_only'],5,frozen)
        manifest_path.write_bytes(original_manifest)
        assert ns['manifest_guard_only'](5,frozen)
    ns={}
    for filename,target in [('imp10_leafwise.py','SETTINGS'),('imp20_tree.py','SETTINGS')]:
        source=ast.parse((ROOT/filename).read_text())
        source.body=[n for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets)]
        if filename=='imp20_tree.py':ns['BASE_SETTINGS']=dict(ns['SETTINGS']);before=dict(ns['SETTINGS'])
        exec(compile(source,filename,'exec'),ns)
    changes={k:[before.get(k),ns['SETTINGS'].get(k)] for k in set(before)|set(ns['SETTINGS']) if before.get(k)!=ns['SETTINGS'].get(k)}
    assert changes=={'num_leaves':[31,63], 'num_threads':[8,4]}
    assert HASHES=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    result=dict(status='PASS',source_sha256=HASHES,valid_baseline_load=True,
                either_day_member_mutation_rejected=mutated_days,audit_hash_mutation_rejected=True,
                unsuccessful_or_unconfirmed_audit_rejected=True,
                changed_manifest_rejected_before_confirmation_setup=True,
                only_parameter_change=changes,
                test_scope='AST baseline loader and first fit guard only; no imports or training.')
    (ARTIFACT_DIR/'imp20_portable_test_baseline_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
