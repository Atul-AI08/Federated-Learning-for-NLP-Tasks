import torch
from torch.nn.functional import softmax
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datetime import datetime
import os

def get_model_checkpoint_name(model):
    timestamp =  datetime.now().strftime("%dd_%mm_%yy_%HH_%MM")
    return model.__class__.__name__ + '_' + timestamp + '.pt'
  
def save_checkpoint(self,model, optimizer, val_loss,val_acc,model_file):

    state_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':val_acc,
                'val_loss': val_loss}
    
    torch.save(state_dict, model_file)
    print(f'Model saved to ==> {model_file}')


def load_checkpoint(self, model,model_file, optimizer=None):


    state_dict = torch.load(model_file)
    print(f'Model loaded from <== {model_file}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    if optimizer is not None :
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['val_loss']


def test(self,model,test_dl,apply_softmax=False):
    
    model.eval()
    test_running_acc = 0
    all_predictions = torch.tensor([])
    y_actual = torch.tensor([])
    tqdm_test_iterator = tqdm(enumerate(test_dl),desc="[TEST]",total=len(test_dl),ascii=True,colour="blue")
    with torch.no_grad():
        for batch_idx,(data,target) in tqdm_test_iterator:
            y_pred = model(data)
            all_predictions = torch.cat((all_predictions,torch.round(y_pred)),dim=0)
            y_actual = torch.cat((y_actual,target),dim=0)
            if apply_softmax:
                # test_running_acc += (self.get_accuracy(softmax(y_pred,dim=1),target))
                test_running_acc += self.get_accuracy_with_softmax(y_pred,target)
            else:
                target = target.unsqueeze(1)
                test_running_acc += self.get_accuracy_without_softmax(y_pred,target)
                
            tqdm_test_iterator.set_postfix(avg_test_acc=f"{test_running_acc/(batch_idx+1):0.4f}")  
            # if (batch_idx+1)%10==0:
            #     print("===")
            #     print(data[0],y_pred[0],target[0])
            #     print("===")
    log = f"Test dataset size {len(test_dl.dataset)}\n"
    log += f"Test accuracy {(test_running_acc/len(test_dl))*100:4f} % "
    
    history = History()
    history.history['name'] = 'Testing History'
    history.history['y_preds'] = all_predictions
    history.history['y_actual'] = y_actual
    history.history['log'] = log

    
    return history 
    
    
def get_accuracy_with_softmax(y_pred,y_actual):
    """Calculates the accuracy (0 to 1)

    Args:
    + y_pred (tensor ): output from the model
    + y_actual (tensor): ground truth 

    Returns:
    + float: a value between 0 to 1
    """
    _, y_pred = torch.max(softmax(y_pred.detach(),dim=1) ,1)
    # print(y_pred,y_actual)
    # print(y_pred.shape,y_actual.shape,torch.sum(y_pred==y_actual),torch.sum(y_pred==y_actual).item())
    return (1/len(y_actual))*torch.sum(y_pred==y_actual)

def get_accuracy_without_softmax(y_pred,y_actual):
    """Calculates the accuracy (0 to 1)

    Args:
    + y_pred (tensor ): output from the model
    + y_actual (tensor): ground truth 

    Returns:
    + float: a value between 0 to 1
    """
    
    # print(y_pred,y_actual)
    # print(y_pred.shape,y_actual.shape,torch.sum(y_pred==y_actual),torch.sum(y_pred==y_actual).item())
    # print(y_pred.shape,y_actual.shape)
    y_pred = torch.argmax(y_pred, axis=1)
    return (1/len(y_actual))*torch.sum(torch.round(y_pred)==y_actual)

def get_num_correct(self,y_pred,y_actual):
    return torch.sum(torch.round(y_pred)==y_actual)
    
    
class History:
    def __init__(self):
        self.history = {}