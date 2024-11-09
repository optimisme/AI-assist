Aquesta carpeta 'test_0':

- script_transform.py tranforma la imatge 'cat.png' en una imatge de grisos que indiquen la profunditat 'cat_depth_gray.png' on el color blanc significa molt proper
- script_server.py posa en funcionament un servidor que rep imatges en format json/base64 i retorna una imatge en format json/base64 amb el mapa de grisos
- script_client.py llegeix un arxiu d'imatge i fa una crida al servidor anterior