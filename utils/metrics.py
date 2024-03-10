import os
import matplotlib.pyplot as plt
import torch
from datetime import datetime
class MetricManager:
    """ 
    Class for tracking accuracy and loss values.
    metrics 
    + 'train_loss_list',
    + 'train_acc_list',
    + 'val_loss_list', 
    + 'val_acc_list',
    + 'global_steps_list'
    
    """
    
    
    def __init__(self,metrics_file):
     
        
        self.global_steps_list = []
        
        self.train_loss_list = []
        self.val_loss_list = []
        
        self.train_acc_list = []
        self.val_acc_list = []
        
        if metrics_file is None:
            metrics_file = 'metrics.pt'
       
        self.metrics_file = metrics_file
        
    def update_train_loss(self,train_loss):
        self.train_loss_list.append(train_loss)
    
    def update_val_loss_list(self,val_loss):
        self.val_loss_list.append(val_loss)
    
    def update_global_steps_list(self,global_steps):
        self.global_steps_list.append(global_steps)
    
    def update(self,train_loss,val_loss,train_acc,val_acc,global_steps):
        self.train_loss_list.append(train_loss)
        self.val_loss_list.append(val_loss)
        self.global_steps_list.append(global_steps)
        self.train_acc_list.append(train_acc)
        self.val_acc_list.append(val_acc)
        
    def get_metrics_string(self):
        return    f"avg_train_loss : {self.train_loss_list[-1]:0.4f}"\
                + f"avg_train_acc : {self.train_acc_list[-1]:0.4f}"\
                + f"avg_val_loss : {self.val_loss_list[-1]:0.4f}"\
                + f"avg_val_acc : {self.val_acc_list[-1]:0.4f}"
                 
    
                 
    def save_metrics(self,file=None):
        if file is None:
            file = self.metrics_file
        metric_state_dict = {'train_loss_list':self.train_loss_list,
                             'train_acc_list':self.train_acc_list,
                             'val_loss_list':self.val_loss_list,
                             'val_acc_list':self.val_acc_list,
                             'global_steps_list':self.global_steps_list}  
                 
                 
        torch.save(metric_state_dict,file)
        print(f"Metrics saved to ==> {file}")
    
   
    def plot_metrics(self):
        """Loads metrics from the file passed during object initialization

        Raises:
            OSError: if given file doesn't exists
        """
        DIR =os.path.dirname(os.path.realpath(__file__))
        # if not os.path.exists(os.path.join(DIR,self.metrics_file)):
        
        
        # if not os.path.exists(self.metrics_file):
        #     raise OSError(f"No file named {self.metrics_file}")
        # else:
            
        metrics_state_dict = torch.load(self.metrics_file)
            
            
        self.global_steps_list = metrics_state_dict['global_steps_list']
        self.train_loss_list = metrics_state_dict['train_loss_list']
        self.val_loss_list = metrics_state_dict['val_loss_list']
        self.train_acc_list= metrics_state_dict['train_acc_list']
        self.val_acc_list= metrics_state_dict['val_acc_list']

        # plt.figure(1)
        # plt.plot(self.global_steps_list, self.train_loss_list, label='Train')
        # plt.plot(self.global_steps_list, self.val_loss_list, label='Valid')
        # plt.xlabel('Global Steps')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.figure(2)
        # plt.plot(self.global_steps_list, self.train_acc_list, label='Train')
        # plt.plot(self.global_steps_list, self.val_acc_list, label='Valid')
        # plt.xlabel('Global Steps')
        # plt.ylabel('Accuracy')
        
        # plt.legend()
        # plt.show() 
        
        fig,(acc_axis,loss_axis) = plt.subplots(nrows=1,ncols=2)
        
        acc_axis.plot(self.global_steps_list, self.train_acc_list, label='Train Accuracy')
        acc_axis.plot(self.global_steps_list, self.val_acc_list, label='Valid Accuracy')
        acc_axis.set_title("Accuracy Curves")
        acc_axis.set(xlabel="Epochs",ylabel="Accuracy")
        acc_axis.legend()
        
        loss_axis.plot(self.global_steps_list, self.train_loss_list, label='Train Loss')
        loss_axis.plot(self.global_steps_list, self.val_loss_list, label='Valid Loss')
        loss_axis.set_title("Loss Curves")
        loss_axis.set(xlabel="Epochs",ylabel="Loss")
        loss_axis.legend()
        
        fig.tight_layout()
        # fig.text(0.5, 0.04, 'Epochs', ha='center')
        # fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
        plt.tight_layout()
        plt.show()
        DIR = os.path.dirname(self.metrics_file)
        fig_name = "plots_"+datetime.now().strftime("%dd_%mm_%yy_%HH_%MM") + ".png"
        fig.savefig(os.path.join(DIR,fig_name))
            
        
def test():
           
    x,y = range(10),range(10)
    fig,(acc_axis,loss_axis) = plt.subplots(nrows=1,ncols=2,sharex=True)
    
    acc_axis.plot(x, y, label='Train Accuracy')
    acc_axis.plot(x, y, label='Valid Accuracy')
    acc_axis.set_title("Accuracy Curves")
    acc_axis.set(ylabel="Accuracy")
    acc_axis.legend()
    
    loss_axis.plot(x, y, label='Train Loss')
    loss_axis.plot(x, y, label='Valid Loss')
    loss_axis.set_title("Loss Curves")
    loss_axis.set(ylabel="Loss")
    loss_axis.legend()
   
    fig.text(0.5, 0.04, 'Epochs', ha='center')
    # fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    fig.tight_layout()
    plt.show()
    fig.savefig("test.png")
    

if __name__ == '__main__':
    test()