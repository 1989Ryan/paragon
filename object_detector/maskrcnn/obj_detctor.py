from matplotlib.font_manager import json_load
import torch
import numpy as np
from object_detector.maskrcnn.mask_rcnn import get_model_instance_segmentation

def read_json(filepath):
    '''
    from filepath to instruction list
    :return:instruction list
    '''
    try:
        import json
        with open(filepath) as f:
            data = json.load(f)
    except IOError as exc:
        raise IOError("%s: %s" % (filepath, exc.strerror))
    return data

class object_detector(object):
    '''mask rcnn object detector'''
    def __init__(self, num_class):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = get_model_instance_segmentation(num_class)
        self.model.to(self.device)
        # self.label2id = json_load
        self.table = read_json('tabletop_gym/envs/config/label2id.json')
    
    def load(self, path):
        '''load model'''
        f1 = torch.load(path, map_location=self.device)
        self.model.load_state_dict(f1)
    
    def pred(self, img):
        '''object detection'''
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(img.to(self.device))
        return prediction
    
    def query(self, img, query_name):
        '''query the bbox given the query name'''
        query_str = query_name.replace("the ", "")
        id = [value for key, value in self.table.items() if query_str in key.lower()]
        prediction = self.pred(img)
        bboxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        mapping = [ele in id for ele in labels]
        query_bboxes = bboxes[mapping]
        query_scores = scores[mapping]
        return query_bboxes, query_scores

