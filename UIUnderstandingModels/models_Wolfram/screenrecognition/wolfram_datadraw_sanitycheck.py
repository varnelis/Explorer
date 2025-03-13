import torchvision
import json
from PIL import ImageDraw,Image
from tqdm import tqdm

def draw_bbox(bbox, img, scale_factor = 1):
    if img is None:
        return 0

    if len(bbox) == 0:
        print('no boxes at confidence >= 0.5 here')

    img_transform = Image.open(img)

    draw_bbox = ImageDraw.Draw(img_transform)
    for i, bbox in enumerate(bbox):
        bbox = bbox["bbox"]
        scaled_bbox = [e * scale_factor for e in bbox[:4]]
        draw_bbox.rectangle(scaled_bbox, outline = "green", width = 3)

    return img_transform

if __name__ == "__main__":
    files = '../../metadata/screenrecognition/test_ids_wolfram.json'
    with open(files,'r') as fp:
        uuids_file = json.load(fp)
    uuids = uuids_file['items']
    
    for uuidset in tqdm(uuids):
        uuid = uuidset["uuid"]
        img = f'../../downloads/wolfram/screenshots/{uuid}.png'
        bboxfile = f'../../downloads/wolfram/annotations/{uuid}.json'
        with open(bboxfile,'r') as fp:
            bboxes = json.load(fp)
        bboxes = bboxes["clickable"]
        imgdrawbbox = draw_bbox(bboxes,img)
        imgdrawbbox.save(f'../../downloads/wolfram/bbox_test_screenshots/{uuid}.png')