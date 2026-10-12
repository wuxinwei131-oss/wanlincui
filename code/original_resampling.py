import numpy as np
import pandas as pd
STATES=("IA","IL","IN","KS","NE","TX")
REGIONS=("1","2","5","6","9")
YEARS=tuple(range(2019,2025))
RNG_SEED=20260913
SEED=20260920
REPS=2000

def resample(b,col,crop):
    # Exact archived RNG ordering: QA2 draws all geography indices then all year
    # indices; TV5 alternates geography/year per draw. np.quantile linear default.
    x=b.pivot(index='geo',columns='year',values=col).sort_index().sort_index(axis=1).to_numpy()
    n=10000 if crop=='corn' else 2000; seed=20260913 if crop=='corn' else 20260920
    rng=np.random.default_rng(seed)
    if crop=='corn':
        gi=rng.integers(0,len(x),(n,len(x))); yi=rng.integers(0,x.shape[1],(n,x.shape[1]))
    else:
        pairs=[(rng.integers(0,len(x),len(x)),rng.integers(0,x.shape[1],x.shape[1])) for _ in range(n)]
        gi,yi=map(np.asarray,zip(*pairs))
    draws=np.array([np.nanmean(x[np.ix_(g,y)]) for g,y in zip(gi,yi)])
    lo,hi=np.quantile(draws[np.isfinite(draws)],[.025,.975])
    return lo,hi,n,seed,draws

def limits(matrix: pd.DataFrame, axis: str, seed: int, repeats: int):
    grid = matrix.unstack("year").reindex(index=STATES if axis == "state" else REGIONS, columns=YEARS).to_numpy(float)
    if not np.isfinite(grid).all():
        raise ValueError("TWO_WAY_GRID_INCOMPLETE")
    rng = np.random.default_rng(seed)
    if axis == "state":
        adraw = rng.integers(0, 6, size=(repeats, 6))
        ydraw = rng.integers(0, 6, size=(repeats, 6))
        draw = grid[adraw[:, :, None], ydraw[:, None, :]].mean(axis=(1, 2))
        valid = repeats
    else:
        # Original TV3A draws regions and years whole, using multiplicity products.
        draw = []
        valid = 0
        for _ in range(repeats):
            r = rng.choice(5, size=5, replace=True)
            y = rng.choice(6, size=6, replace=True)
            rc = np.bincount(r, minlength=5)
            yc = np.bincount(y, minlength=6)
            w = rc[:, None] * yc[None, :]
            if w.sum():
                draw.append(float(np.sum(grid*w) / w.sum()))
                valid += 1
    return float(np.percentile(draw, 2.5)), float(np.percentile(draw, 97.5)), valid

def key_columns(condition: str):
    return ["ERS_region", "year"] if condition == "ROUTE_A2_EXTERNAL_POST_DISCLOSURE" else ["state", "year"]

def paired_two_way(blocks: pd.DataFrame, condition: str, left: str, right: str, field: str = "PB3") -> dict:
    group = key_columns(condition)
    a = blocks.loc[blocks.condition.eq(condition) & blocks.policy.eq(left), group + [field]].rename(columns={field: "left"})
    b = blocks.loc[blocks.condition.eq(condition) & blocks.policy.eq(right), group + [field]].rename(columns={field: "right"})
    paired = a.merge(b, on=group, validate="one_to_one")
    matrix = paired.pivot(index=group[0], columns=group[1], values="left").to_numpy(float) - paired.pivot(index=group[0], columns=group[1], values="right").to_numpy(float)
    rng = np.random.default_rng(RNG_SEED)
    a_idx = rng.integers(0, matrix.shape[0], size=(10000, matrix.shape[0]))
    b_idx = rng.integers(0, matrix.shape[1], size=(10000, matrix.shape[1]))
    res = np.asarray([matrix[np.ix_(a_idx[i], b_idx[i])].mean() for i in range(10000)])
    return {"condition": condition, "left": left, "right": right, "metric": field,
            "paired_blocks": len(paired), "observed_left_minus_right": float(matrix.mean()),
            "two_way_2p5": float(np.quantile(res, .025)), "two_way_97p5": float(np.quantile(res, .975)),
            "descriptive_resample_P_left_better": float(np.mean(res < 0)), "resamples": 10000, "seed": RNG_SEED}

def two_way_descriptive(frame: pd.DataFrame, value: str, group: str = "block") -> dict:
    f = frame[[group, "year", value]].dropna().copy()
    groups = sorted(f[group].unique())
    years = sorted(f.year.unique())
    point = float(f[value].mean()) if len(f) else None
    if len(groups) < 2 or len(years) < 2:
        return {"point": point, "lower_2p5": None, "upper_97p5": None,
                "group_clusters": len(groups), "year_clusters": len(years), "blocks": len(f),
                "method": "NOT_ESTIMABLE_FEWER_THAN_TWO_WHOLE_CLUSTERS"}
    table = f.pivot(index=group, columns="year", values=value).reindex(index=groups, columns=years).to_numpy(float)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(REPS):
        gi = rng.integers(0, len(groups), size=len(groups))
        yi = rng.integers(0, len(years), size=len(years))
        sample = table[np.ix_(gi, yi)]
        if np.isfinite(sample).any():
            boot.append(float(np.nanmean(sample)))
    return {"point": point, "lower_2p5": float(np.quantile(boot, .025)) if boot else None,
            "upper_97p5": float(np.quantile(boot, .975)) if boot else None,
            "group_clusters": len(groups), "year_clusters": len(years), "blocks": len(f),
            "method": "PAIRED_TWO_WAY_WHOLE_GROUP_AND_YEAR_DESCRIPTIVE_RESAMPLING",
            "repetitions": REPS, "seed": SEED, "county_iid_inference": False}
