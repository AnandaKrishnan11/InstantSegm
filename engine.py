import torch
from torch import nn

@torch.no_grad()
def test_all(model, metric, test_loader, device, num_class):
    model.eval()
    out_list = []
    has_labels = test_loader.dataset.masks is not None

    for i, data in enumerate(test_loader):
        x, y = data
        x = x.to(device)
        out = model(x, with_probs=True)

        if num_class == 1:
            out = (out >= 0.5).long().squeeze(1)  
        else:
            out = torch.argmax(out, dim=1)         

        out_list.append(out.squeeze(0).cpu().numpy()) 

        if has_labels:
            y = y.to(device)
            metric.compute_metrics(ref=y.squeeze(1), pred=out)

    metrics = metric.get_metrics() if has_labels else None
    return metrics, out_list
