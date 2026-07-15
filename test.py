import torch
from model import create_model
import argparse
from data.loader import createdataset
from data.data_utils import mosaicChips
from utils import Accuracy
from engine import test_all

def main(args):
    
    test_loader = createdataset(img_path= args.source_path, mask_path= args.mask_path)
    meta = test_loader.dataset.meta
    device = "cpu"
    model = create_model()
    model.load_state_dict(torch.load("/app/checkpoint/best_weight.pt",map_location="cpu", weights_only=True))
    model = model.to(device=device)

    metric = Accuracy()
    test_metric,out = test_all(model=model, metric=metric, test_loader=test_loader, device=device, num_class=1)
    print(test_metric)
    mosaicChips(out,meta,save_path=args.save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help='Source path to the s1 scene')
    parser.add_argument("--mask_path", type=str, required=False, help='Source path to the mask')
    parser.add_argument("--save_path", type=str, required=True, help='full path to the water mask with file extension')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args=args)
