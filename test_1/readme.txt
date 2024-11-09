Aquesta carpeta 'test_1':

- script_transform.py tranforma la imatge 'cat.png' en una imatge de grisos que indiquen la profunditat 'cat_depth_gray.png' on el color blanc significa molt proper
  a més, mostra informació dels objectes detectats a la imatge i la seva posició, mida i profunditat (al punt central de l'objecte)
- script_server.py posa en funcionament un servidor que rep imatges en format json/base64 i retorna la informació anterior
- script_client.py llegeix un arxiu d'imatge i fa una crida al servidor anterior