"""Public IO adapter. Not part of the original scientific algorithms."""
from pathlib import Path
import json,os
PACKAGE=Path(__file__).resolve().parents[1]
def method_path(stage):
    if stage=='qa1_code':return PACKAGE/'methods'
    return Path(os.environ.get('SR_METHOD_DATA_ROOT',str(PACKAGE/'external_inputs')))/stage
def method_config(name):
    return json.loads((PACKAGE/'config/method_parameters.json').read_text(encoding='utf-8'))[name]
def earth_engine_project():
    return os.environ['SR_EARTH_ENGINE_PROJECT']
