#IMPORTS
import argparse
import time
from tqdm import tqdm
import copy as cp
#torch
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader, DataListLoader
#plot
import matplotlib
from matplotlib import pyplot as plt
#local scripts
from utils.data_loader import *
from utils.eval_helper import *
from center_loss import CenterLoss

#MODEL
class Model(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model = args.model
        self.concat = concat
        # convolutional layer model
        if self.model == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)
        # concatenation
        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)            
        # last layer
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)      
    def forward(self, data):
        # assigning
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        #convolutional layer
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = gmp(x, batch)
        # concatenation (graphs + content)
        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))
        # classification
        y = F.log_softmax(self.lin2(x), dim=-1)
        #return
        return self.lin2(x), y

#ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')
# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=40, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='gcn', help='model type, [gcn, gat, sage]')
# additional parameters
parser.add_argument('--loss', type=str, default='merged', help='Type of center loss [merged, unmerged, none]')
parser.add_argument('--splits', type=list, default=[.2,.1], help='dataset split [train, val], must be a list of percentages')
parser.add_argument('--alpha', type=float, default=.5, help='weight for center loss')
parser.add_argument('--lr_cent', type=float, default=.01, help='learning rate for center loss')
#parse
args, _ = parser.parse_known_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed) 

#DATASET
dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
print(args)

#LOADING DATA
num_training = int(len(dataset) * args.splits[0])
num_val = int(len(dataset) * args.splits[1])
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
# type of device
if args.multi_gpu:
    loader = DataListLoader
else:
    loader = DataLoader
# loading splits
train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

#MODEL INSTANTIATION
model = Model(args, concat=args.concat)
if args.multi_gpu:
    model = DataParallel(model)
model = model.to(args.device)

#OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if not args.loss == 'none':
    center_loss = CenterLoss(num_classes=args.num_classes, feat_dim=args.num_classes, use_gpu=args.device=='gpu')
# merge
    if args.loss == 'merged':
        params = list(model.parameters()) + list(center_loss.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.loss == 'unmerged':
        optimizer_cl = torch.optim.Adam(center_loss.parameters(), lr=args.lr_cent, weight_decay=args.weight_decay)
    
#COMPUTE TEST FUNCTION
@torch.no_grad()
def compute_test(loader, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
    plot_x = []
    plot_y = []
    for data in loader:
        if not args.multi_gpu:
            data = data.to(args.device)
        features, out = model(data)
        plot_x.append(features)
        plot_y.append(data.y)
        if args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
        else:
            y = data.y
        if verbose:
            print(F.softmax(out, dim=1).cpu().numpy())
        out_log.append([F.softmax(out, dim=1), y])
        loss_test += F.nll_loss(out, y).item()  
    return eval_deep(out_log, loader), loss_test, np.concatenate(plot_x,0), np.concatenate(plot_y,0)

#VISUALIZATION
def plot_features(features, labels, name):
    colors = ['C0', 'C1']
    plt.figure(figsize=(8,6))
    for label_idx in range(2):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1)
    plt.legend(['0', '1'], loc='upper right')
    plt.savefig('figures/' + name + '.png')
    plt.show()
    plt.close()
    
#MAIN
if __name__ == '__main__':
    # Model training
    min_loss = 1e10
    val_loss_values = []
    best_epoch = 0
    t = time.time()
    model.train()
    plot_train_x = [] # train
    plot_train_y = []
    plot_val_x = [] # val
    plot_val_y = []
    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        out_log = []
        plot_train_x_temporary = []
        plot_train_y_temporary = []
        for i, data in enumerate(train_loader):
            if not args.multi_gpu:
                data = data.to(args.device)
            features, out = model(data)
            plot_train_x_temporary.append(features.data.numpy())
            plot_train_y_temporary.append(data.y.data.numpy())
            if args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
            else:
                y = data.y
            loss = F.nll_loss(out, y)
            if not args.loss == 'none':
                loss += center_loss(features, y) * args.alpha
                if args.loss == 'unmerged':
                    optimizer_cl.zero_grad()
            optimizer.zero_grad()          
            loss.backward()
            optimizer.step()
            #start of center update
            if not args.loss == 'none':
                for param in center_loss.parameters():
                    if args.loss == 'merged':
                        param.grad.data *= (args.lr_cent / (args.alpha * args.lr))                    
                    elif args.loss == 'unmerged':
                        param.grad.data *= (1./args.alpha)
                        optimizer_cl.step()                                
            #end of center update 
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
        plot_train_x.append(np.concatenate(plot_train_x_temporary,0))
        plot_train_y.append(np.concatenate(plot_train_y_temporary,0))        
        acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val, plot_v_x, plot_v_y = compute_test(val_loader)
        plot_val_x.append(plot_v_x)
        plot_val_y.append(plot_v_y)        
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
              f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
              f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
              f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')
    # testset
    [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss, plot_test_x, plot_test_y = compute_test(test_loader)
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
          f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
    
#PLOTTING
merged = {'rx' : plot_train_x,
          'ry' : plot_train_y,
          'vx' : plot_val_x,
          'vy' : plot_val_y,
          'tx' : plot_test_x,
          'ty' : plot_test_y}
epoch_pl = 39
plot_features(merged['rx'][epoch_pl], merged['ry'][epoch_pl], 'merged_train')
plot_features(merged['tx'], merged['ty'], 'merged_test')