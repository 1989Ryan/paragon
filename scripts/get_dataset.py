import os
import gdown

train_0 = 'https://drive.google.com/uc?export=download&id=1WxfGi-CfmvQODyQgTkrAfCSQHPmNRMsE'
train_1 = 'https://drive.google.com/uc?export=download&id=1Jj56r4k9_kWBRba7TytIRZLvoPHdbqm4'
train_2 = 'https://drive.google.com/uc?export=download&id=1MlnFa1PYJCEhSYeqUeJtrJz35_Ad5u7c'
test_0 = 'https://drive.google.com/uc?export=download&id=1OchYHD8NLkec79Tu062nmAX1DrLHN2-G'
test_1 = 'https://drive.google.com/uc?export=download&id=1avAONbv-xK893jo2ZXW40moIiKOnRAZt'
output_0 = 'train_4_obj_nvisii.zip'
output_1 = 'train_10_obj_nvisii.zip'
output_2 = 'train_11_obj_nvisii.zip'
output_3 = 'test_10_obj_nvisii.zip'
output_4 = 'test_11_obj_nvisii.zip'

gdown.download(train_0, output_0, quiet=False)
os.system(f'unzip {output_0} -d ./dataset')
os.system(f'rm {output_0}')

gdown.download(train_1, output_1, quiet=False)
os.system(f'unzip {output_1} -d ./dataset')
os.system(f'rm {output_1}')

gdown.download(train_2, output_2, quiet=False)
os.system(f'unzip {output_2} -d ./dataset')
os.system(f'rm {output_2}')

gdown.download(test_0, output_3, quiet=False)
os.system(f'unzip {output_3} -d ./dataset')
os.system(f'rm {output_3}')

gdown.download(test_1, output_4, quiet=False)
os.system(f'unzip {output_4} -d ./dataset')
os.system(f'rm {output_4}')



