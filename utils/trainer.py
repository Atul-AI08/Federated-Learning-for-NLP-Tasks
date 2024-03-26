# Description: This file contains the training and testing functions for the model.

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_accuracy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Train the model
def fit(
    net,
    train_dataloader,
    epochs,
    lr,
    args_optimizer,
    device="cpu",
):
    net = net.to(device)
    
    # Set optimizer
    if args_optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5)
    elif args_optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5, amsgrad=True
        )
    elif args_optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=1e-5
        )

    # Set loss function
    criterion = nn.CrossEntropyLoss()
    net.train()

    for epoch in range(epochs):
        print()
        train_running_loss = 0
        train_running_acc = 0   
        
        tqdm_train_iterator = tqdm(enumerate(train_dataloader),
                                    desc=f"[train]{epoch+1}/{epochs}",
                                    ascii=True,leave=True,
                                    total=len(train_dataloader),
                                    colour="green",position=0)
        
        # Training loop
        for batch_idx,(data,target) in tqdm_train_iterator:

            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_pred = net(data)

            loss = criterion(y_pred,target)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            
            train_running_acc += get_accuracy(y_pred.detach(),target)
                
            tqdm_train_iterator.set_postfix(avg_train_acc=f"{train_running_acc/(batch_idx+1):0.4f}",
                                            avg_train_loss=f"{(train_running_loss/(batch_idx+1)):0.4f}")

    net.eval()
    print(" ** Training complete **")

# Test the model
def test(
    net,
    test_dl,
    args,
    verbose=True,
    device="cpu"
):
    net = net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    tqdm_train_iterator = tqdm(enumerate(test_dl),
                                desc="[TEST]",
                                total=len(test_dl),
                                ascii=True,leave=True,
                                colour="green",position=0)
    
    test_running_loss = 0
    test_running_acc = 0

    actuals = []
    predictions = []

    for idx, (data, target) in tqdm_train_iterator:
        data = data.to(device)
        target = target.to(device)
        y_pred = net(data)
        loss = criterion(y_pred,target)

        test_running_loss += loss.item()

        actuals.extend(target.cpu().numpy())
        predictions.extend(y_pred.argmax(dim=1).cpu().numpy())
        
        test_running_acc += get_accuracy(y_pred.detach(),target)
            
        tqdm_train_iterator.set_postfix(avg_test_acc=f"{test_running_acc/(idx+1):0.4f}",
                                        avg_test_loss=f"{(test_running_loss/(idx+1)):0.4f}")
        
    print("Test Loss: ", test_running_loss/len(test_dl))

    # Classification Report
    if verbose:
        print("Metrics Report:")
        print(classification_report(actuals, predictions))

    # Confusion Matrix
    cfm = confusion_matrix(actuals, predictions)
    classes = ['Negative', 'Positive']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cfm, annot=True, cmap="Blues", fmt="d", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.savefig(f"plots/cm_{args.dataset}_{args.partition}.png", format="png", dpi=300)