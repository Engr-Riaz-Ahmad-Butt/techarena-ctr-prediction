"""Alignment and fixed-blend invariants; no fitting or score claims."""
import json
from pathlib import Path
import numpy as np
import runner
import choice_xgb_ensemble as ensemble


def main():
    rng = np.random.default_rng(6803)
    shared = dict(y=np.tile([1,0,0,0,0,0,0,0,0,0], 10),
                  events=np.repeat(np.arange(10), 10),
                  baseline=rng.random(100), weights=np.ones(100))
    tree = dict(shared, raw=rng.random(100))
    neural = dict(shared, raw=rng.random(100))
    actual = ensemble.combine(tree, neural, runner)
    expected = runner.rank(.8*runner.rank(shared['baseline'])
                           + .1*runner.rank(tree['raw']) + .1*runner.rank(neural['raw']))
    assert np.array_equal(actual, expected)
    for key in ['y', 'events', 'baseline', 'weights']:
        bad = dict(neural); bad[key] = neural[key].copy(); bad[key][0] += 1
        try: ensemble.combine(tree, bad, runner)
        except AssertionError: pass
        else: raise AssertionError('Accepted misalignment: '+key)
    bad = dict(neural); bad['raw'] = neural['raw'].copy(); bad['raw'][0] = np.nan
    try: ensemble.combine(tree, bad, runner)
    except AssertionError: pass
    else: raise AssertionError('Accepted non-finite prediction')
    report = dict(status='PASS', checks=['exact fixed blend', 'four alignment guards',
                                        'finite probability guard'], fitting=False)
    path = Path(__file__).resolve().parents[1]/'results/choice_xgb_ensemble_test.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
