"""Label-free feature drift diagnostic on deterministic samples of causal caches."""
import json
from pathlib import Path
import numpy as np
from build_bundle import feature_paths


def sample(name, day, count=20000):
    parts=[]
    for path in feature_paths(name, day):
        x=np.load(path,mmap_mode='r')
        index=np.linspace(0,len(x)-1,min(count,len(x)),dtype=int)
        parts.append(np.asarray(x[index],dtype='float64'))
        x._mmap.close()
    return np.column_stack(parts)


def main():
    reports=[]
    for day in [5,6]:
        training=np.concatenate([sample(f'day{d}',d) for d in range(day-3,day)])
        validation=sample(f'val{day}',day)
        mean=training.mean(0);std=training.std(0);constant=std<1e-8
        shift=np.abs(validation.mean(0)-mean)/np.maximum(std,.01)
        rows=[]
        for col in np.argsort(shift)[-15:][::-1]:
            rows.append(dict(column=int(col),training_mean=float(mean[col]),
                             training_std=float(std[col]),validation_mean=float(validation[:,col].mean()),
                             standardized_mean_shift=float(shift[col]),fitting_constant=bool(constant[col])))
        reports.append(dict(day=day,training_rows=len(training),validation_rows=len(validation),
                            fitting_constant_columns=np.flatnonzero(constant).tolist(),
                            most_shifted=rows,shift_over_three_sd_columns=np.flatnonzero(shift>3).tolist()))
    root=Path(__file__).resolve().parents[1]
    destination=root/'results/tabular_choice_label_free_shift.json'
    destination.write_text(json.dumps(dict(reports=reports,
        scope='No click labels read. Deterministic samples, descriptive covariate drift only; not score evidence.'),indent=2)+'\n')
    print(json.dumps(reports))


if __name__=='__main__':main()
