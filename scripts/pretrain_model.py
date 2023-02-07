import os
import gdown

url = 'https://drive.google.com/uc?export=download&id=1biQDBCUVELdYLm1NVydZbocMrJSt2THO'
output = 'paragon_pretrain.tar.gz'

gdown.download(url, output, quiet=False)
if not os.path.isdir('./tmp'):
    os.mkdir('./tmp')
os.system(f'tar -xv {output} -C ./tmp')
os.system(f'rm {output}')


url = 'https://drive.google.com/uc?export=download&id=1li3Y1M_aSblB77HZ_d4qX3wumDJNMYdl'
output = 'maskrcnn_pretrain.tar.gz'

gdown.download(url, output, quiet=False)
os.system(f'tar -xv {output} -C ./object_detector/maskrcnn/')
os.system(f'rm {output}')