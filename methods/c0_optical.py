from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import hashlib
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine

ROOT = method_path('corn_predictors')
B0=ROOT.parent/"CRP_LOCAL_PRODUCT_GRID_B0_20260912_133000"
AUTH = method_config("02_TARGET_GRID_AUTHORITY.json")["MOD09A1"]
BANDS=AUTH["output_bands"]
TILE=512
ENDPOINTS={("IA",2012),("TX",2012),("IL",2024),("KS",2024)}
EVENT_LOCK=Lock()


def dates(year:int)->list[str]:
    source=ee.ImageCollection("MODIS/061/MOD09A1").filterDate(f"{year}-04-01",f"{year}-09-01")
    indices=sorted(source.aggregate_array("system:index").getInfo())
    eligible=[]
    for index in indices:
        start=date.fromisoformat(index.replace("_","-"))
        end=start+timedelta(days=7)
        if date(year,4,1)<=start and end<=date(year,8,31):
            eligible.append(index)
    if not eligible or len(set(eligible))!=len(eligible):
        raise RuntimeError(f"Optical date registry unavailable {year}")
    (ROOT/f"C0_OPTICAL_DATES_{year}.json").write_text(json.dumps({
        "collection":"MODIS/061/MOD09A1","year":year,"all_indices":indices,
        "eligible_indices":eligible,"rule":"start >= April 1 and start + 7 days <= August 31",
        "eligible_count":len(eligible)},indent=2),encoding="utf-8")
    return eligible


def prepared(index:str)->ee.Image:
    image=ee.Image(ee.ImageCollection("MODIS/061/MOD09A1").filter(ee.Filter.eq("system:index",index)).first())
    qa=image.select("QA")
    state=image.select("StateQA")
    mask=qa.bitwiseAnd(3).eq(0)
    for i in range(7):
        mask=mask.And(qa.rightShift(2+4*i).bitwiseAnd(15).eq(0))
    mask=(mask.And(state.bitwiseAnd(3).eq(0))
          .And(state.rightShift(2).bitwiseAnd(1).eq(0))
          .And(state.rightShift(3).bitwiseAnd(7).eq(1))
          .And(state.rightShift(6).bitwiseAnd(3).lte(2))
          .And(state.rightShift(8).bitwiseAnd(3).lte(1))
          .And(state.rightShift(10).bitwiseAnd(1).eq(0))
          .And(state.rightShift(11).bitwiseAnd(1).eq(0))
          .And(state.rightShift(12).bitwiseAnd(1).eq(0))
          .And(state.rightShift(13).bitwiseAnd(1).eq(0))
          .And(state.rightShift(15).bitwiseAnd(1).eq(0)))
    refl=image.select([f"sur_refl_b0{i}" for i in range(1,8)],BANDS[:7]).multiply(.0001).updateMask(mask)
    red=refl.select("red_reflectance")
    nir=refl.select("nir_reflectance")
    swir=refl.select("swir_1640nm_reflectance")
    ndvi=nir.subtract(red).divide(nir.add(red)).rename("ndvi")
    evi2=nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(2.4)).add(1)).rename("evi2")
    nirv=ndvi.multiply(nir).rename("nirv")
    ndmi=nir.subtract(swir).divide(nir.add(swir)).rename("ndmi")
    values=refl.addBands([ndvi,evi2,nirv,ndmi])
    valid=values.mask().reduce(ee.Reducer.min()).rename("product_valid").unmask(0).toFloat()
    return values.unmask(-9999).addBands(valid)


def event(obj:dict)->None:
    with EVENT_LOCK:
        with (ROOT/"C0_OPTICAL_TRANSPORT_EVENTS.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"logged_utc":datetime.now(timezone.utc).isoformat(),**obj})+"\n")


def prefetch_tile(image:ee.Image, state:str, year:int, image_index:str, grid:dict,
                  base:Affine, tr:int, tc:int)->int:
    height=min(TILE,grid["height"]-tr)
    width=min(TILE,grid["width"]-tc)
    tile=f"r{tr:05d}_c{tc:05d}.tif"
    path=ROOT/"raw"/"mod09"/state/str(year)/image_index/tile
    transform=base*Affine.translation(tc,tr)
    if path.exists():
        return 0
    path.parent.mkdir(parents=True,exist_ok=True)
    request={"expression":image,"fileFormat":"GEO_TIFF","bandIds":BANDS+["product_valid"],
             "grid":{"dimensions":{"width":width,"height":height},
                     "affineTransform":{"scaleX":transform.a,"shearX":transform.b,
                                        "translateX":transform.c,"shearY":transform.d,
                                        "scaleY":transform.e,"translateY":transform.f},
                     "crsCode":AUTH["crs"]},"workloadTag":"crp_c0_sparse_native_optical"}
    retries=0
    started=time.monotonic()
    while True:
        try:
            blob=ee.data.computePixels(request)
            break
        except Exception as exc:
            retries+=1
            if retries>2:
                event({"status":"FAILED","state":state,"year":year,"index":image_index,
                       "tile":tile,"error_type":type(exc).__name__,"error":str(exc),"retries":retries})
                raise
            time.sleep(min(5*retries,10))
    with MemoryFile(blob) as mem:
        with mem.open() as ds:
            if (ds.width,ds.height,ds.count)!=(width,height,len(BANDS)+1):
                raise RuntimeError(f"Prefetched optical tile dimension mismatch {state} {year} {tile}")
            if max(abs(a-b) for a,b in zip(list(ds.transform)[:6],list(transform)[:6]))>1e-5:
                raise RuntimeError(f"Prefetched optical native affine mismatch {state} {year} {tile}")
    path.write_bytes(blob)
    event({"status":"COMPLETE","state":state,"year":year,"index":image_index,
           "source_image_id":f"MODIS/061/MOD09A1/{image_index}","tile":tile,
           "bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest(),
           "retries":retries,"runtime_seconds":time.monotonic()-started,
           "global_row0":grid["global_row0"]+tr,"global_col0":grid["global_col0"]+tc})
    return 1


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--state",required=True)
    parser.add_argument("--year",type=int,required=True)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    state,year=args.state,args.year
    if not 1<=args.workers<=4:
        raise RuntimeError("Bounded optical prefetch workers must be 1..4")
    if (state,year) not in ENDPOINTS and not (ROOT/"C0_ENDPOINT_PASS.json").exists():
        raise RuntimeError("Stage 2 prohibited before endpoint pass")
    ee.Initialize(project=earth_engine_project())
    assert ee.Number(1).getInfo()==1
    grid=json.loads((ROOT/"raw"/"weights"/f"{state}_{year}_MOD09A1_grid.json").read_text(encoding="utf-8"))
    weights=pd.read_parquet(ROOT/"raw"/"weights"/f"{state}_{year}_MOD09A1_weights.parquet")
    counties=json.loads((ROOT/"raw"/"counties"/f"ee_tiger_2018_{state}_counties.geojson").read_text(encoding="utf-8"))["features"]
    ids=[f["properties"]["GEOID"] for f in counties]
    index={key:i for i,key in enumerate(ids)}
    if len(index)!=len(ids):
        raise RuntimeError("County FIPS not unique")
    weights["county_index"]=weights["county_fips"].map(index)
    if weights["county_index"].isna().any():
        raise RuntimeError("Weight county missing in geometry")
    weights["local_row"]=weights["grid_row"]-grid["global_row0"]
    weights["local_col"]=weights["grid_col"]-grid["global_col0"]
    weights["tile_row"]=(weights["local_row"]//TILE)*TILE
    weights["tile_col"]=(weights["local_col"]//TILE)*TILE
    groups={(int(r),int(c)):part for (r,c),part in weights.groupby(["tile_row","tile_col"],sort=True)}
    if not groups or any((part["nominal_weight_q255"]<=0).any() for part in groups.values()):
        raise RuntimeError("Sparse window support error")
    nominal=np.bincount(weights["county_index"].to_numpy(int),weights=weights["nominal_weight_q255"],minlength=len(ids))
    indices=dates(year)
    base=Affine(*grid["transform"])
    rows=[]
    transfer_bytes=0
    request_count=0
    began=time.monotonic()
    began_cpu=time.process_time()
    for image_index in indices:
        start=date.fromisoformat(image_index.replace("_","-"))
        if start<date(year,4,1) or start+timedelta(days=7)>date(year,8,31):
            raise RuntimeError("Optical cutoff violation")
        image=prepared(image_index)
        valid_weight=np.zeros(len(ids),dtype="float64")
        numerators=np.zeros((len(BANDS),len(ids)),dtype="float64")
        if args.workers>1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                request_count+=sum(pool.map(lambda pair:prefetch_tile(image,state,year,image_index,grid,base,*pair),groups))
        for (tr,tc),frame in groups.items():
            height=min(TILE,grid["height"]-tr)
            width=min(TILE,grid["width"]-tc)
            tile=f"r{tr:05d}_c{tc:05d}.tif"
            path=ROOT/"raw"/"mod09"/state/str(year)/image_index/tile
            transform=base*Affine.translation(tc,tr)
            if path.exists():
                blob=path.read_bytes()
                with MemoryFile(blob) as mem:
                    with mem.open() as ds:
                        if (ds.width,ds.height,ds.count)!=(width,height,len(BANDS)+1):
                            raise RuntimeError(f"Existing optical tile dimension mismatch {state} {year} {tile}")
                        if max(abs(a-b) for a,b in zip(list(ds.transform)[:6],list(transform)[:6]))>1e-5:
                            raise RuntimeError(f"Existing optical tile native affine mismatch {state} {year} {tile}")
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                request={"expression":image,"fileFormat":"GEO_TIFF","bandIds":BANDS+["product_valid"],
                         "grid":{"dimensions":{"width":width,"height":height},
                                 "affineTransform":{"scaleX":transform.a,"shearX":transform.b,
                                                    "translateX":transform.c,"shearY":transform.d,
                                                    "scaleY":transform.e,"translateY":transform.f},
                                 "crsCode":AUTH["crs"]},"workloadTag":"crp_c0_sparse_native_optical"}
                retry=0
                tile_start=time.monotonic()
                while True:
                    try:
                        blob=ee.data.computePixels(request)
                        request_count+=1
                        break
                    except Exception as exc:
                        retry+=1
                        if retry>2:
                            event({"status":"FAILED","state":state,"year":year,"index":image_index,
                                   "tile":tile,"error_type":type(exc).__name__,"error":str(exc),"retries":retry})
                            raise
                        time.sleep(min(5*retry,10))
                with MemoryFile(blob) as mem:
                    with mem.open() as ds:
                        if (ds.width,ds.height,ds.count)!=(width,height,len(BANDS)+1):
                            raise RuntimeError(f"Optical tile dimension mismatch {state} {year} {tile}")
                        if max(abs(a-b) for a,b in zip(list(ds.transform)[:6],list(transform)[:6]))>1e-5:
                            raise RuntimeError(f"Optical tile native affine mismatch {state} {year} {tile}")
                path.write_bytes(blob)
                event({"status":"COMPLETE","state":state,"year":year,"index":image_index,
                       "source_image_id":f"MODIS/061/MOD09A1/{image_index}","tile":tile,
                       "bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest(),
                       "retries":retry,"runtime_seconds":time.monotonic()-tile_start,
                       "global_row0":grid["global_row0"]+tr,"global_col0":grid["global_col0"]+tc})
            transfer_bytes+=len(blob)
            with MemoryFile(blob) as mem:
                with mem.open() as ds:
                    values=ds.read()
            r=frame["local_row"].to_numpy(int)-tr
            c=frame["local_col"].to_numpy(int)-tc
            pixels=values[:,r,c]
            valid=(pixels[-1]>.5)&np.isfinite(pixels[:-1]).all(axis=0)
            valid&=(pixels[:-1]!=-9999).all(axis=0)
            w=frame["nominal_weight_q255"].to_numpy(float)
            ci=frame["county_index"].to_numpy(int)
            vw=np.where(valid,w,0.)
            valid_weight+=np.bincount(ci,weights=vw,minlength=len(ids))
            for k in range(len(BANDS)):
                numerators[k]+=np.bincount(ci,weights=np.where(valid,w*pixels[k],0),minlength=len(ids))
        if np.any(valid_weight>nominal*(1+1e-8)):
            raise RuntimeError("Valid optical crop weight exceeds nominal")
        for i,fips in enumerate(ids):
            row={"state":state,"county_fips":fips,"target_year":year,
                 "composite_start":str(start),"composite_end":str(start+timedelta(days=7)),
                 "source_image_id":f"MODIS/061/MOD09A1/{image_index}",
                 "nominal_crop_weight_m2":float(nominal[i]),"valid_crop_weight_m2":float(valid_weight[i]),
                 "valid_weight_fraction":float(valid_weight[i]/nominal[i]) if nominal[i]>0 else math.nan,
                 "effective_prior_corn_area_acres":float(nominal[i]/4046.8564224)}
            for k,name in enumerate(BANDS):
                row[name]=float(numerators[k,i]/valid_weight[i]) if valid_weight[i]>0 else math.nan
            rows.append(row)
        print(f"OPTICAL {state} {year} {image_index} valid_counties={int((valid_weight>0).sum())} bytes={transfer_bytes}",flush=True)
    out=ROOT/"state_year"/f"{state}_{year}_optical.parquet"
    out.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    if len(frame)!=len(ids)*len(indices) or frame.duplicated(["county_fips","target_year","composite_start"]).any():
        raise RuntimeError("Optical county-composite row identity mismatch")
    frame.to_parquet(out,index=False,compression="zstd")
    metrics={"state":state,"target_year":year,"county_count":len(ids),"eligible_composites":len(indices),
             "optical_rows":len(frame),"sparse_window_count":len(groups),"transfer_bytes":transfer_bytes,
             "compute_pixels_requests":request_count,"prefetch_workers":args.workers,
             "runtime_seconds":time.monotonic()-began,
             "local_cpu_seconds":time.process_time()-began_cpu,
             "peak_ram_bytes":psutil.Process().memory_info().peak_wset,"parquet_sha256":hashlib.sha256(out.read_bytes()).hexdigest()}
    (ROOT/"state_year"/f"{state}_{year}_optical_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"OPTICAL_DONE {state} {year} {len(frame)} {transfer_bytes}",flush=True)


if __name__=="__main__":
    main()
