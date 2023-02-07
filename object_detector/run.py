from object_detector import maskrcnn
from object_detector.space.eval.ap import convert_to_boxes
from object_detector.maskrcnn.mask_rcnn import get_model_instance_segmentation
import torch

# detector = {
#     'space': Space,
#     'maskrcnn': maskrcnn,
# }

# class objDetector():
#     def __init__(self, model_name, device) -> None:
#         self.model = detector[model_name]
#         self.model.to(device)
#         self.device = device

#     @torch.no_grad()
#     def run(self, img):
#         boxes_pred = []
#         self.model.eval() 
#         img.to(self.device)
#         loss, log = self.model(img, global_step=10000000000)
#         z_where, z_pres_prob = log['z_shere'], log['z_pres_prob']
#         z_where = z_where.detach().cpu()
#         z_pres_prob = z_pres_prob.detach().cpu().squeeze()
#         z_pres = z_pres_prob > 0.5
#         boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob)
#         boxes_pred.extend(boxes_batch)
#         return boxes_pred
    
class maskrcnn_obj_detecotr():
    '''mask rcnn object detector'''
    def __init__(self, path, num_class=2):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = get_model_instance_segmentation(num_class)
        self.model.to(self.device)
        self.load(path)

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
    
    def query(self, img):
        '''query the bbox given the query name'''
        prediction = self.pred(img)
        bboxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        return bboxes, scores