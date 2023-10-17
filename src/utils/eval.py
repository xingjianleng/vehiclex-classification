import torch
from sklearn.metrics import precision_recall_fscore_support


def evaluate_model(model, test_loader):
    # helper function to evaluate a model on a given dataset
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for inputs, targets in test_loader:
            labels.extend(targets.cpu().numpy())
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)
            _, predict = torch.max(logits, 1)
            preds.extend(predict.cpu().numpy())
            correct += (predict == targets).sum().cpu().item()
            total += targets.size(0)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = correct / total
    return acc * 100, precision * 100, recall * 100, f1 * 100
