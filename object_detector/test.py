import torch
from object_detector.run import maskrcnn_obj_detecotr
from paragon.tabletop_dataset import tabletop_gym_obj_dataset
import matplotlib.pyplot as plt
from matplotlib import patches
import torchvision.transforms as T

def visualize(bboxes, img):
    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(img)
    for bbox in bboxes:
        # plt.text(bbox[0], bbox[3], str(v[order[counter]].item()))
        rect = patches.Rectangle((bbox[0], bbox[3]), bbox[2]-bbox[0], bbox[1]-bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.savefig("visualize.png")
    input()

def eval_main():
    dataset = tabletop_gym_obj_dataset('/home/zirui/tabletop_gym/dataset/test_4_obj_cliport', test=True)
    data_n = len(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0)
    obj_detect = maskrcnn_obj_detecotr('/home/zirui/paraground/trained_model/0_8_mask_rcnn.pt')
    for i, data in enumerate(data_loader, 0):
        # bboxes = list(data['bbox_dict'].values())
        img = data['image']
        # print(img[0])
        trans = T.ToPILImage()
        noise = torch.abs(torch.randn(img.size())* 0.1)
        img = img + noise
        bboxes, scores = obj_detect.query(img)
        print(bboxes)
        img = trans(img[0]) 
        visualize(bboxes, img)

if __name__ == '__main__':
    eval_main()    