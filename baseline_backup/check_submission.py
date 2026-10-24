"""Independent, strict submission format and coverage check."""
from config import TRAIN_CSV, EVALUATION_CSV
import csv, hashlib, json, math, sys
from pathlib import Path

from config import ARTIFACT_DIR as root
submission=Path(sys.argv[1]) if len(sys.argv)>1 else root/'submission_v2.csv'
source=EVALUATION_CSV
with source.open(newline='',encoding='utf-8') as f:
    expected={r['rowid'] for r in csv.DictReader(f)}
seen=set();count=0;lo=1.;hi=0.;total=0.
with submission.open(newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f)
    assert reader.fieldnames==['rowid','predicted_probability'], 'Wrong header'
    for row in reader:
        assert set(row)=={'rowid','predicted_probability'}, 'Extra fields'
        key=row['rowid']
        assert key in expected, 'Unknown row ID'
        assert key not in seen, 'Duplicate row ID'
        p=float(row['predicted_probability'])
        assert math.isfinite(p) and 0<=p<=1, 'Invalid probability'
        seen.add(key);count+=1;lo=min(lo,p);hi=max(hi,p);total+=p
assert seen==expected, 'Missing evaluation rows'
assert count==529390
report={'file':str(submission),'rows':count,'missing':0,'unknown':0,'duplicates':0,
        'invalid_probabilities':0,'minimum':lo,'maximum':hi,'mean':total/count,
        'sha256':hashlib.file_digest(submission.open('rb'),'sha256').hexdigest()}
submission.with_suffix('.checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
