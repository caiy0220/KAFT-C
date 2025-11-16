from utils import basic 
from utils.basic import log
import torch, time
from datetime import timedelta
from tqdm import tqdm
from torch import nn
from torch.amp import autocast, GradScaler

# Validation
def val_model(m, dl, pbar=None):
    criterion = torch.nn.CrossEntropyLoss()
    m.eval()
    device = basic.get_device(m)
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        count = 0
        for inputs, labels in dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = m(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            count += 1

            if pbar:
                pbar.set_postfix_str(f"Current val: [{count}/{len(dl)}]")

    val_acc = correct / total
    val_loss /= len(dl)
    return val_acc, val_loss


def train(m, tdl, vdl, epochs=100, save_pth=None, scaler=None, 
          scheduler=None, optimizer=None, criterion=None):
    """
    m:          [torch.model] Target model to train
    tdl:        [DataLoader] Training set
    vdl:        [DataLoader] Validation set
    epochs:     [int] #epochs
    save_pth:   [str] Best model on validation set will be saved to 'save_pth'
                      No saving if set to None
    scaler:     [TODO] Flag to activating AMP intended for training acceleration
    scheduler:  [] Scheduler for learning rate, optional
    optimizer:  []
    criterion:  []
    """
    device = basic.get_device(m)
    if optimizer is None:
        optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_acc = 0.
    batch_per_epoch = len(tdl)

    tik = time.time()   # Timer for total training cost
    log()  
    msg = ''
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            m.train()
            total_loss, correct, total = 0, 0, 0
            epoch_tik = time.time() # Timer for training cost per epoch

            # Standard training loopA
            count = 0
            for inputs, labels in tdl:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with autocast(device.type):
                    outputs = m(inputs)
                    loss = criterion(outputs, labels)

                # accelerating with AMP if enabled
                if scaler:
                    scaler(loss, optimizer, parameters=m.parameters())
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                labels = labels 
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
                count += 1
                pbar.set_postfix_str(f'{count}/{len(tdl)} | {msg}')

            pbar.update(1)

            # Collecting training & validation status 
            train_acc = correct / total
            train_loss = total_loss / batch_per_epoch
            train_losses.append(train_loss)
            val_acc, val_loss = val_model(m, vdl)
            val_losses.append(val_loss)
        
            # epoch_cost = time.time() - epoch_tik
            # _, cost_str = get_eta(epoch_cost, epochs-epoch+1)
            scheduler.step(epoch)

            if (save_pth and val_acc > best_acc):
                torch.save(m.state_dict(), save_pth)
                best_acc = val_acc

            msg = f"LR: {optimizer.param_groups[0]['lr']:.6f} Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}, Best Acc: {best_acc*100:.2f}",

    tok = time.time()
    cost = str(timedelta(seconds=int(tok - tik)))
    log(f'Total training time: {cost}')
    return m, train_losses, val_losses, len(tdl)

def get_eta(epoch_cost, num_epochs):
    eta = epoch_cost * num_epochs
    cost_str = str(timedelta(seconds=int(epoch_cost)))
    eta_str = str(timedelta(seconds=int(eta)))
    return eta_str, cost_str
