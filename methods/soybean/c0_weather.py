from __future__ import annotations
import sys
from pathlib import Path as _PublicPath
sys.path.insert(0,str(_PublicPath(__file__).resolve().parents[1]))
from _paths import method_path, method_config, earth_engine_project

import argparse
import calendar
import hashlib
import json
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine

ROOT = method_path('soybean/predictor')
BASE=["mean_tmax_c","mean_tmin_c","total_precip_mm","mean_vpd_kpa",
      "mean_shortwave_w_m2","total_eto_mm","water_balance_mm","gdd_base10_cap30_cdays",
      "hot_days_gt30_c","hot_days_gt35_c","excess_degree_sum_gt30_c"]
ENDPOINTS={('IA',2019),('AR',2024)}


def monthly_image(year:int,month:int)->tuple[ee.Image,list[str],list[str]]:
    days=calendar.monthrange(year,month)[1]
    start=f"{year}-{month:02d}-01"
    end=f"{year+1}-01-01" if month==12 else f"{year}-{month+1:02d}-01"
    source=ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(start,end).sort("system:time_start")
    indices=source.aggregate_array("system:index").getInfo()
    if len(indices)!=days:
        raise RuntimeError(f"gridMET missing daily source {year} {month}: {len(indices)}/{days}")
    pr=source.select("pr")
    tmax=source.select("tmmx").map(lambda im:ee.Image(im).subtract(273.15))
    tmin=source.select("tmmn").map(lambda im:ee.Image(im).subtract(273.15))
    eto=source.select("eto")
    gdd=source.map(lambda im:ee.Image(im).select("tmmx").add(ee.Image(im).select("tmmn"))
                   .divide(2).subtract(273.15).clamp(10,30).subtract(10))
    hot30=tmax.map(lambda im:ee.Image(im).gt(30).toFloat())
    hot35=tmax.map(lambda im:ee.Image(im).gt(35).toFloat())
    excess=tmax.map(lambda im:ee.Image(im).subtract(30).max(0))
    summary=ee.Image.cat([
        tmax.mean().rename(BASE[0]),tmin.mean().rename(BASE[1]),
        pr.sum().rename(BASE[2]),source.select("vpd").mean().rename(BASE[3]),
        source.select("srad").mean().rename(BASE[4]),eto.sum().rename(BASE[5]),
        pr.sum().subtract(eto.sum()).rename(BASE[6]),gdd.sum().rename(BASE[7]),
        hot30.sum().rename(BASE[8]),hot35.sum().rename(BASE[9]),excess.sum().rename(BASE[10])])
    daily_names=[f"pr_day_{i:02d}" for i in range(1,days+1)]
    daily=pr.toBands().rename(daily_names)
    values=summary.addBands(daily)
    valid=values.mask().reduce(ee.Reducer.min()).rename("product_valid").unmask(0).toFloat()
    return values.unmask(-9999).addBands(valid),BASE+daily_names+["product_valid"],indices


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--state",required=True)
    parser.add_argument("--year",type=int,required=True)
    args=parser.parse_args()
    state,year=args.state,args.year
    if (state,year) not in ENDPOINTS and not (ROOT/"C0_ENDPOINT_PASS.json").exists():
        raise RuntimeError("Stage 2 prohibited before endpoint pass")
    ee.Initialize(project=earth_engine_project())
    assert ee.Number(1).getInfo()==1
    grid=json.loads((ROOT/"raw"/"weights"/f"{state}_{year}_GRIDMET_grid.json").read_text(encoding="utf-8"))
    weights=pd.read_parquet(ROOT/"raw"/"weights"/f"{state}_{year}_GRIDMET_weights.parquet")
    counties=json.loads((ROOT/"raw"/"counties"/f"ee_tiger_2018_{state}_counties.geojson").read_text(encoding="utf-8"))["features"]
    ids=[f["properties"]["GEOID"] for f in counties]
    id_index={key:i for i,key in enumerate(ids)}
    ci=weights["county_fips"].map(id_index).to_numpy(int)
    rr=(weights["grid_row"]-grid["global_row0"]).to_numpy(int)
    cc=(weights["grid_col"]-grid["global_col0"]).to_numpy(int)
    ww=weights["nominal_weight_q255"].to_numpy(float)
    nominal=np.bincount(ci,weights=ww,minlength=len(ids))
    t=Affine(*grid["transform"])
    out={key:{"state":state,"county_fips":key,"target_year":year,
              "nominal_crop_weight_m2":float(nominal[i]),
              "effective_prior_soybean_area_acres":float(nominal[i]/4046.8564224)} for i,key in enumerate(ids)}
    daily_precip={key:[] for key in ids}
    events=[]
    transferred=0
    requests=0
    began=time.monotonic()
    began_cpu=time.process_time()
    for month in range(1,9):
        image,bands,indices=monthly_image(year,month)
        path=ROOT/"raw"/"gridmet"/state/str(year)/f"{month:02d}_summary_daily_pr.tif"
        if path.exists():
            blob=path.read_bytes()
            with MemoryFile(blob) as mem:
                with mem.open() as ds:
                    if (ds.width,ds.height,ds.count)!=(grid["width"],grid["height"],len(bands)):
                        raise RuntimeError("Existing gridMET tile dimensions mismatch")
                    if max(abs(a-b) for a,b in zip(list(ds.transform)[:6],list(t)[:6]))>1e-9:
                        raise RuntimeError("Existing gridMET native affine mismatch")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            request={"expression":image,"fileFormat":"GEO_TIFF","bandIds":bands,
                     "grid":{"dimensions":{"width":grid["width"],"height":grid["height"]},
                             "affineTransform":{"scaleX":t.a,"shearX":t.b,"translateX":t.c,
                                                "shearY":t.d,"scaleY":t.e,"translateY":t.f},
                             "crsCode":"EPSG:4326"},"workloadTag":"crp_c0_native_gridmet"}
            retries=0
            tile_start=time.monotonic()
            while True:
                try:
                    blob=ee.data.computePixels(request)
                    requests+=1
                    break
                except Exception as exc:
                    retries+=1
                    if retries>2:
                        raise
                    time.sleep(min(5*retries,10))
            events.append({"state":state,"target_year":year,"month":month,"status":"COMPLETE",
                           "bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest(),
                           "retries":retries,"runtime_seconds":time.monotonic()-tile_start,
                           "date_start":f"{year}-{month:02d}-01",
                           "date_end":f"{year}-{month:02d}-{calendar.monthrange(year,month)[1]:02d}",
                           "source_daily_count":len(indices)})
            with (ROOT/"C0_WEATHER_TRANSPORT_EVENTS.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(events[-1])+"\n")
            with MemoryFile(blob) as mem:
                with mem.open() as ds:
                    if (ds.width,ds.height,ds.count)!=(grid["width"],grid["height"],len(bands)):
                        raise RuntimeError("gridMET native tile dimensions mismatch")
                    if max(abs(a-b) for a,b in zip(list(ds.transform)[:6],list(t)[:6]))>1e-9:
                        raise RuntimeError("gridMET native affine mismatch")
            path.write_bytes(blob)
        transferred+=len(blob)
        with MemoryFile(blob) as mem:
            with mem.open() as ds:
                pixels=ds.read()[:,rr,cc]
        valid=(pixels[-1]>.5)&np.isfinite(pixels[:-1]).all(axis=0)
        valid&=(pixels[:-1]!=-9999).all(axis=0)
        vw=np.where(valid,ww,0.)
        valid_w=np.bincount(ci,weights=vw,minlength=len(ids))
        if np.any(valid_w>nominal*(1+1e-8)):
            raise RuntimeError("gridMET valid weight exceeds nominal")
        for j,key in enumerate(ids):
            out[key][f"m{month:02d}_valid_weight_fraction"]=float(valid_w[j]/nominal[j]) if nominal[j]>0 else math.nan
        for k,name in enumerate(bands[:-1]):
            numer=np.bincount(ci,weights=np.where(valid,pixels[k]*ww,0.),minlength=len(ids))
            mean=np.divide(numer,valid_w,out=np.full(len(ids),np.nan),where=valid_w>0)
            if name.startswith("pr_day_"):
                for j,key in enumerate(ids):
                    daily_precip[key].append(float(mean[j]))
            else:
                for j,key in enumerate(ids):
                    out[key][f"m{month:02d}_{name}"]=float(mean[j])
        print(f"WEATHER {state} {year} month={month} valid_counties={int((valid_w>0).sum())} bytes={transferred}",flush=True)
    for key in ids:
        row=out[key]
        for name in ("total_precip_mm","total_eto_mm","water_balance_mm","gdd_base10_cap30_cdays",
                     "hot_days_gt30_c","hot_days_gt35_c","excess_degree_sum_gt30_c"):
            values=[row[f"m{m:02d}_{name}"] for m in range(1,9)]
            row[f"jan_aug_{name}"]=float(sum(values)) if all(math.isfinite(x) for x in values) else math.nan
        longest=run=0
        complete=True
        for pr in daily_precip[key]:
            if not math.isfinite(pr):
                complete=False
                run=0
            elif pr<1.0:
                run+=1
                longest=max(longest,run)
            else:
                run=0
        row["longest_dry_spell_days_pr_lt1mm"]=longest if complete else math.nan
        row["gridmet_daily_precip_complete"]=complete
        row["gridmet_date_start"]=f"{year}-01-01"
        row["gridmet_date_end"]=f"{year}-08-31"
    frame=pd.DataFrame([out[key] for key in ids])
    if len(frame)!=len(ids) or frame.duplicated(["county_fips","target_year"]).any():
        raise RuntimeError("Weather county-year row identity mismatch")
    outpath=ROOT/"state_year"/f"{state}_{year}_weather.parquet"
    outpath.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(outpath,index=False,compression="zstd")
    metrics={"state":state,"target_year":year,"county_count":len(ids),"weather_rows":len(frame),
             "daily_source_count":sum(calendar.monthrange(year,m)[1] for m in range(1,9)),
             "date_start":f"{year}-01-01","date_end":f"{year}-08-31",
             "transfer_bytes":transferred,"compute_pixels_requests":requests,
             "runtime_seconds":time.monotonic()-began,"local_cpu_seconds":time.process_time()-began_cpu,
             "peak_ram_bytes":psutil.Process().memory_info().peak_wset,
             "parquet_sha256":hashlib.sha256(outpath.read_bytes()).hexdigest()}
    (ROOT/"state_year"/f"{state}_{year}_weather_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"WEATHER_DONE {state} {year} {len(frame)} {transferred}",flush=True)


if __name__=="__main__":
    main()
