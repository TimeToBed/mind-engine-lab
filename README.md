# mind-engine-lab (Python)

Researcher-facing Mind Engine sandbox:

1) Import world config Excel (Object_Table_world_config_v0_4.xlsx)
2) Generate/load Observation (offline, reproducible)
3) Configure ActionMeaning (DSL) + choose built-in plugins
4) Compile to YAML
5) Test with Observation -> L2/L3/L4 outputs + trace

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## WebUI

```bash
streamlit run app.py
```

## CLI

```bash
mind-engine import-world --xlsx /path/to/Object_Table_world_config_v0_4.xlsx --out data/world.json
mind-engine gen-obs --world data/world.json --out data/observations.json --n 20 --seed 42
mind-engine init-config --out data/config.yaml
mind-engine test --config data/config.yaml --world data/world.json --obs data/observations.json --index 0
```
