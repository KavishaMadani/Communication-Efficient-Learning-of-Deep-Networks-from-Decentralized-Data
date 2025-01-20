import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_accuracy(loader, model):
  model.eval()

  total = 0
  correct = 0
  with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # To calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  return accuracy


def calculate_precision(model, dataloader, device=torch.device("cpu")):
    """
    Calculate precision for a trained PyTorch model.
    
    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to perform computations on (default: CPU).
    
    Returns:
        float: Precision value.
    """
    model.eval()  # Set model to evaluation mode
    true_positives = 0
    false_positives = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)  # Assumes multi-class classification
            
            # Calculate true positives and false positives
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
    
    # Precision = TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    return precision


precision = calculate_precision(model, dataloader, device=torch.device("cuda"))
print(f"Precision: {precision:.4f}")

def calculate_recall(model, dataloader, device=torch.device("cpu")):
    """
    Calculate recall for a trained PyTorch model.
    
    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to perform computations on (default: CPU).
    
    Returns:
        float: Recall value.
    """
    model.eval()  # Set model to evaluation mode
    true_positives = 0
    false_negatives = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)  # Assumes multi-class classification
            
            # Calculate true positives and false negatives
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
    
    # Recall = TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    return recall


recall = calculate_recall(model, dataloader, device=torch.device("cuda"))
print(f"Recall: {recall:.4f}")

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# Example usage
precision = calculate_precision(model, dataloader, device=torch.device("cuda"))
recall = calculate_recall(model, dataloader, device=torch.device("cuda"))
f1_score = calculate_f1_score(precision, recall)

print(f"F1 Score: {f1_score:.4f}")

'''
def calculate_accuracy_and_loss(loader, model, criterion):
  model.eval()

  total = 0
  correct = 0
  running_loss = 0.0
  with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # To calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # To calculate loss
        loss = criterion(outputs, labels)  # Accumulate testing loss
        running_loss += loss.item()

  accuracy = 100 * correct / total
  loss = running_loss / len(loader)
  return accuracy, loss
'''