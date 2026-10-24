from config import TRAIN_CSV, EVALUATION_CSV
import csv, json, time
from pathlib import Path
import numpy as np

from config import ARTIFACT_DIR as ROOT

users, items = {}, {}
def encode(mapping, key):
    if key not in mapping: mapping[key] = len(mapping)
    return mapping[key]

def convert(path, name, count, labelled):
    data = np.lib.format.open_memmap(ROOT / (name + '.npy'), mode='w+', dtype='int32', shape=(count, 7))
    histories = np.lib.format.open_memmap(ROOT / (name + '_hist.npy'), mode='w+', dtype='int32', shape=(count // 10, 20))
    histories[:] = -1
    previous = None
    start = time.time()
    with path.open(newline='', encoding='utf-8') as f:
        for n, row in enumerate(csv.DictReader(f)):
            event = n // 10
            uid, iid = encode(users, row['user_id']), encode(items, row['item_id'])
            if n % 10 == 0:
                previous = row['event_id']
                h = [encode(items, x) for x in json.loads(row['history_item_ids'])]
                assert len(h) <= 20
                histories[event, :len(h)] = h
            assert previous == row['event_id'], 'Unexpected event ordering'
            data[n] = uid, iid, int(row['timestamp']), int(row['time_bucket']), int(row['day_type']), int(row['clicked']) if labelled else -1, event
            if (n+1) % 1000000 == 0: print(name, n+1, 'rows', round(time.time()-start), 'seconds', flush=True)
    assert n+1 == count
    data.flush(); histories.flush()
    print(name, 'complete', n+1, flush=True)

convert(TRAIN_CSV, 'train', 6868240, True)
convert(EVALUATION_CSV, 'evaluation', 529390, False)
(ROOT / 'ids.json').write_text(json.dumps({'users': users, 'items': items}))
print('Users:', len(users), 'Items:', len(items), flush=True)
