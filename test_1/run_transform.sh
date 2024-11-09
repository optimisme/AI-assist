#!/bin/bash
python3.10 -m venv ../depth_map_env
source ../depth_map_env/bin/activate
python3.10 script_transform.py
deactivate
