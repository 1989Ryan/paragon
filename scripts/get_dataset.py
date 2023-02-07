import os
import gdown

train_0 = 'https://drive.google.com/file/d/1WxfGi-CfmvQODyQgTkrAfCSQHPmNRMsE/view?usp=sharing'
train_1 = 'https://drive.google.com/file/d/1Jj56r4k9_kWBRba7TytIRZLvoPHdbqm4/view?usp=sharing'
train_2 = 'https://drive.google.com/file/d/1MlnFa1PYJCEhSYeqUeJtrJz35_Ad5u7c/view?usp=sharing'
test_0 = 'https://drive.google.com/file/d/1OchYHD8NLkec79Tu062nmAX1DrLHN2-G/view?usp=sharing'
test_1 = 'https://drive.google.com/file/d/1avAONbv-xK893jo2ZXW40moIiKOnRAZt/view?usp=sharing'
output_model = 'paragon_pretrain.tar.gz'

gdown.download(train_0, output_model, quiet=False)
os.system(f'unzip {output_model}')
os.system(f'rm {output_model}')