import os
import gdown

url = 'https://drive.google.com/file/d/1biQDBCUVELdYLm1NVydZbocMrJSt2THO/view?usp=sharing'
output = 'paragon_pretrain.tar.gz'

gdown.download(url, output, quiet=False)
os.system(f'tar -xvz {output}')
os.system(f'rm {output}')