#!/bin/bash

# Crea l'entorn virtual amb Python 3.10
python3.10 -m venv depth_map_env
source depth_map_env/bin/activate

# Actualitza pip i instal·la les dependències dins l'entorn virtual
pip install torch torchvision pillow matplotlib timm flask

# Desactiva l'entorn després de la instal·lació
deactivate
