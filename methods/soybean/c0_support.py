from __future__ import annotations
import sys
from pathlib import Path as _PublicPath
sys.path.insert(0,str(_PublicPath(__file__).resolve().parents[1]))
from _paths import method_path, method_config, earth_engine_project

import argparse
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box, shape
from shapely.ops import transform as shape_transform, unary_union
from shapely.strtree import STRtree

ROOT = method_path('soybean/predictor')
ORIGINAL_C0 = Path(str(method_path('corn_predictors')))
B0 = Path(str(method_path('grid_authority')))
B1 = Path(str(method_path('weather_grid_authority')))
AUTH = method_config("02_TARGET_GRID_AUTHORITY.json")
GRIDMET_BASE = method_config("B1_GRIDMET_GRID_AUTHORITY.json")["global_transform"]
MODIS_CRS = CRS.from_proj4(AUTH["local_crs_proj4"])
GEOD = Geod(ellps="WGS84")
ACRE_M2 = 4046.8564224
TILE = 512
ENDPOINTS={('IA',2019),('AR',2024)}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def counties(state: str) -> list[tuple[str, object]]:
    path = ROOT / "raw" / "counties" / f"ee_tiger_2018_{state}_counties.geojson"
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = sorted((f["properties"]["GEOID"], shape(f["geometry"])) for f in data["features"])
    if len(pairs) != len(set(k for k, _ in pairs)) or any(not g.is_valid for _, g in pairs):
        raise RuntimeError(f"County geometry identity/validity failed {state}")
    return pairs


def grid(state: str, product: str, pairs: list[tuple[str, object]]) -> dict:
    if product == "MOD09A1":
        base = Affine(*AUTH["MOD09A1"]["transform"])
        tr = Transformer.from_crs(4326, MODIS_CRS, always_xy=True).transform
        geoms = [shape_transform(tr, g) for _, g in pairs]
        crs = AUTH["local_crs_proj4"]
    else:
        base = Affine(*GRIDMET_BASE)
        geoms = [g for _, g in pairs]
        crs = "EPSG:4326"
    bounds = [g.bounds for g in geoms]
    minx, miny = min(b[0] for b in bounds), min(b[1] for b in bounds)
    maxx, maxy = max(b[2] for b in bounds), max(b[3] for b in bounds)
    col0 = math.floor((minx-base.c)/base.a)-1
    col1 = math.ceil((maxx-base.c)/base.a)+1
    row0 = math.floor((base.f-maxy)/-base.e)-1
    row1 = math.ceil((base.f-miny)/-base.e)+1
    t = base*Affine.translation(col0,row0)
    return {"state":state,"product":product,"global_row0":row0,"global_col0":col0,
            "height":row1-row0,"width":col1-col0,"transform":list(t)[:6],"crs":crs,
            "county_geometry_sha256":sha(ROOT/"raw"/"counties"/f"ee_tiger_2018_{state}_counties.geojson")}


def archive(year: int) -> Path:
    if 2018 <= year <= 2020:
        path = B0 / "raw" / f"{year}_30m_cdls.zip"
    else:
        path = ORIGINAL_C0 / "raw" / "cdl" / f"{year}_30m_cdls.zip"
    if not path.exists():
        raise RuntimeError(f"Official CDL archive missing {year}")
    return path


def extracted(year: int) -> Path:
    if 2018 <= year <= 2020:
        path = B0 / "raw" / f"cdl_{year}_extracted" / f"{year}_30m_cdls.tif"
        if not path.exists():
            raise RuntimeError(f"Frozen B0 source TIFF missing {year}")
        return path
    path = Path(str(method_path('corn_external_predictors'))) / "raw" / "staged" / f"{year}_30m_cdls.tif"
    if path.exists():
        return path
    raise RuntimeError("Read-only TV3A staged CDL source missing; do not write authority")
    with zipfile.ZipFile(archive(year)) as z:
        members = [x for x in z.infolist() if x.filename.lower().endswith((".tif", ".img"))]
        if len(members) != 1 or Path(members[0].filename).name != f"{year}_30m_cdls.tif":
            raise RuntimeError(f"Official CDL raster member mismatch {year}")
        member = members[0]
        part = path.with_suffix(".tif.part")
        if part.exists():
            raise RuntimeError(f"Inspect partial extraction before retry {part}")
        with z.open(member) as src, part.open("xb") as dst:
            while b := src.read(8*1024*1024):
                dst.write(b)
        if part.stat().st_size != member.file_size:
            raise RuntimeError(f"Extracted length mismatch {year}")
        part.rename(path)
    print(f"EXTRACTED {year} {path.stat().st_size}",flush=True)
    return path


def binary(year: int, state: str, pairs: list[tuple[str, object]]) -> Path:
    path = ROOT / "raw" / "binary" / f"{state}_{year}_soybean_valid.tif"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    source = extracted(year)
    with rasterio.open(source) as src:
        if src.count < 1:
            raise RuntimeError(f"CDL raster metadata unusable {year}")
        tr = Transformer.from_crs(4326, src.crs, always_xy=True).transform
        bounds = [shape_transform(tr,g).bounds for _,g in pairs]
        minx, miny = min(b[0] for b in bounds)-1000, min(b[1] for b in bounds)-1000
        maxx, maxy = max(b[2] for b in bounds)+1000, max(b[3] for b in bounds)+1000
        inverse = ~src.transform
        c0,r0 = inverse*(minx,maxy)
        c1,r1 = inverse*(maxx,miny)
        left,top = max(0,math.floor(c0)),max(0,math.floor(r0))
        right,bottom = min(src.width,math.ceil(c1)),min(src.height,math.ceil(r1))
        window = Window(left,top,right-left,bottom-top)
        profile = src.profile.copy()
        profile.update(driver="GTiff",count=2,dtype="uint8",width=int(window.width),
                       height=int(window.height),transform=window_transform(window,src.transform),
                       nodata=255,tiled=True,blockxsize=512,blockysize=512,
                       compress="DEFLATE",predictor=2,BIGTIFF="IF_SAFER")
        part=path.with_suffix(".tif.part")
        if part.exists():
            raise RuntimeError(f"Inspect partial binary before retry {part}")
        with rasterio.open(part,"w",**profile) as dst:
            for _,dw in dst.block_windows(1):
                sw=Window(left+dw.col_off,top+dw.row_off,dw.width,dw.height)
                value=src.read(1,window=sw)
                valid=src.read_masks(1,window=sw)>0
                dst.write(np.where(valid,value==5,255).astype("uint8"),1,window=dw)
                dst.write(valid.astype("uint8"),2,window=dw)
        part.rename(path)
        meta={"year":year,"archive":str(archive(year)),"archive_sha256":sha(archive(year)),
              "source_crs":src.crs.to_wkt(),"source_transform":list(src.transform)[:6],
              "source_dimensions":[src.width,src.height],"source_dtype":src.dtypes[0],
              "source_nodata":src.nodata,"soybean_class":5,"binary_path":str(path),
              "binary_sha256":sha(path)}
        (ROOT / f"C0_CDL_SOURCE_{state}_{year}.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"BINARY {state} {year} {path.stat().st_size}",flush=True)
    return path


def fraction(path: Path, g: dict) -> np.ndarray:
    t=Affine(*g["transform"])
    out=np.full((g["height"],g["width"]),np.nan,dtype="float32")
    valid=np.zeros_like(out)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src,1),destination=out,src_transform=src.transform,
                  src_crs=src.crs,src_nodata=255,dst_transform=t,dst_crs=g["crs"],
                  dst_nodata=np.nan,resampling=Resampling.average,num_threads=4,warp_mem_limit=512)
        reproject(source=rasterio.band(src,2),destination=valid,src_transform=src.transform,
                  src_crs=src.crs,src_nodata=255,dst_transform=t,dst_crs=g["crs"],
                  dst_nodata=0,resampling=Resampling.average,num_threads=4,warp_mem_limit=512)
    valid[~np.isfinite(out)]=0
    if np.any((valid<0)|(valid>1)):
        raise RuntimeError("Warp coverage fraction outside [0,1]")
    return np.where((valid>0)&np.isfinite(out),out,np.nan)


def boundary(geom):
    if geom.geom_type in ("Polygon","MultiPolygon"):
        return geom.boundary
    if hasattr(geom,"geoms"):
        return unary_union([x for g in geom.geoms if (x:=boundary(g)) is not None and not x.is_empty])
    return None


def pixel_areas(g: dict) -> np.ndarray:
    t=Affine(*g["transform"])
    area=np.empty(g["height"],dtype="float64")
    if g["product"] == "GRIDMET":
        x=t.c+(g["width"]//2)*t.a
        for row in range(g["height"]):
            y=t.f+row*t.e
            area[row]=abs(GEOD.polygon_area_perimeter([x,x+t.a,x+t.a,x],[y,y,y+t.e,y+t.e])[0])
    else:
        to_geo=Transformer.from_crs(MODIS_CRS,4326,always_xy=True)
        x=t.c+(g["width"]//2)*t.a
        for row in range(g["height"]):
            y=t.f+row*t.e
            lon,lat=to_geo.transform([x,x+t.a,x+t.a,x],[y,y,y+t.e,y+t.e])
            area[row]=abs(GEOD.polygon_area_perimeter(lon,lat)[0])
    return area


def weights(state: str, year: int, product: str, pairs: list[tuple[str, object]]) -> dict:
    started=time.monotonic()
    started_cpu=time.process_time()
    g=grid(state,product,pairs)
    years=(year-3,year-2,year-1)
    if any(y>=year or y<2009 or y>2023 for y in years):
        raise RuntimeError("CDL chronology violation")
    paths=[binary(y,state,pairs) for y in years]
    fractions=[fraction(path,g) for path in paths]
    all_valid=np.isfinite(fractions).all(axis=0)
    p=np.where(all_valid,(np.nan_to_num(fractions[0])+np.nan_to_num(fractions[1])+np.nan_to_num(fractions[2]))/3,0).astype("float32")
    rows,cols=np.nonzero(p>0)
    if product=="MOD09A1":
        tr=Transformer.from_crs(4326,MODIS_CRS,always_xy=True).transform
        geoms=[shape_transform(tr,geom) for _,geom in pairs]
    else:
        geoms=[geom for _,geom in pairs]
    fips=np.array([key for key,_ in pairs],dtype=object)
    t=Affine(*g["transform"])
    labels=rasterize(((geom,i+1) for i,geom in enumerate(geoms)),out_shape=p.shape,
                     transform=t,fill=0,dtype="uint16",all_touched=False)
    edges=rasterize(((boundary(geom),1) for geom in geoms),out_shape=p.shape,
                    transform=t,fill=0,dtype="uint8",all_touched=True)
    areas=pixel_areas(g)
    inside=(edges[rows,cols]==0)&(labels[rows,cols]>0)
    ir,ic=rows[inside],cols[inside]
    records=[pd.DataFrame({"grid_row":ir+g["global_row0"],"grid_col":ic+g["global_col0"],
                            "county_fips":fips[labels[ir,ic]-1],"p":p[ir,ic],
                            "target_pixel_area_m2":areas[ir],"overlap_q255":np.ones(len(ir))})]
    tree=STRtree(geoms)
    planar=abs(t.a*t.e)
    edge_records=[]
    for r,c in zip(rows[~inside],cols[~inside]):
        x=t.c+int(c)*t.a
        y=t.f+int(r)*t.e
        cell=box(x,y+t.e,x+t.a,y)
        for i in tree.query(cell):
            overlap=geoms[i].intersection(cell).area/planar
            if overlap>0:
                q=round(min(1.,max(0.,overlap))*255)/255
                edge_records.append((int(r)+g["global_row0"],int(c)+g["global_col0"],
                                     fips[i],float(p[r,c]),float(areas[r]),q))
    records.append(pd.DataFrame(edge_records,columns=records[0].columns))
    frame=pd.concat(records,ignore_index=True)
    frame["nominal_weight_q255"]=frame["p"]*frame["target_pixel_area_m2"]*frame["overlap_q255"]
    frame=frame[frame["nominal_weight_q255"]>0].copy()
    frame.sort_values(["county_fips","grid_row","grid_col"],inplace=True)
    out=ROOT/"raw"/"weights"/f"{state}_{year}_{product}_weights.parquet"
    out.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(out,index=False,compression="zstd")
    grid_path=ROOT/"raw"/"weights"/f"{state}_{year}_{product}_grid.json"
    grid_path.write_text(json.dumps(g,indent=2),encoding="utf-8")
    expected_windows=math.ceil(g["height"]/TILE)*math.ceil(g["width"]/TILE)
    lr=frame["grid_row"].to_numpy()-g["global_row0"]
    lc=frame["grid_col"].to_numpy()-g["global_col0"]
    sparse_windows=len(set(zip((lr//TILE).tolist(),(lc//TILE).tolist())))
    result={"state":state,"target_year":year,"product":product,"prior_cdl_years":years,
            "county_count":len(pairs),"positive_support_cells":int(len(rows)),
            "cell_county_rows":len(frame),"all_windows":expected_windows,"sparse_windows":sparse_windows,
            "skipped_empty_windows":expected_windows-sparse_windows,"weight_sum_m2":float(frame["nominal_weight_q255"].sum()),
            "grid":g,"weights_path":str(out),"weights_sha256":sha(out),
            "peak_ram_bytes":psutil.Process().memory_info().peak_wset,
            "runtime_seconds":time.monotonic()-started,"local_cpu_seconds":time.process_time()-started_cpu}
    (ROOT/"raw"/"weights"/f"{state}_{year}_{product}_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"WEIGHTS {state} {year} {product} cells={len(rows)} rows={len(frame)} windows={sparse_windows}/{expected_windows}",flush=True)
    return result


def main() -> None:
    started=time.monotonic()
    started_cpu=time.process_time()
    parser=argparse.ArgumentParser()
    parser.add_argument("--state",required=True,choices=('AL', 'AR', 'DE', 'IA', 'IL', 'KS', 'KY', 'LA', 'MD', 'MI', 'MN', 'MO', 'MS', 'NC', 'ND', 'NE', 'OH', 'OK', 'PA', 'SC', 'SD', 'TN', 'TX', 'VA', 'WI'))
    parser.add_argument("--year",required=True,type=int)
    args=parser.parse_args()
    if (args.state,args.year) not in ENDPOINTS and not (ROOT/"C0_ENDPOINT_PASS.json").exists():
        raise RuntimeError("Non-endpoint state-year prohibited until Stage 1 pass")
    pairs=counties(args.state)
    for product in ("MOD09A1","GRIDMET"):
        weights(args.state,args.year,product,pairs)
    (ROOT/"raw"/"weights"/f"{args.state}_{args.year}_runtime.json").write_text(json.dumps({
        "state":args.state,"target_year":args.year,"wall_seconds":time.monotonic()-started,
        "local_cpu_seconds":time.process_time()-started_cpu,
        "peak_ram_bytes":psutil.Process().memory_info().peak_wset},indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
