from torch.utils.data import TensorDataset
from torch.optim import Adam
import math
import torch
from torch.utils.data import DataLoader
from torch.nn import Parameter

class MyLittleSLALOM(torch.nn.Module):
    def __init__(self, my_tokens, device="cpu"):
        super().__init__()
        self.device= device
        self.indexer = torch.zeros(torch.max(my_tokens)+1, dtype=torch.long).to(device)
        self.indexer[my_tokens] = torch.arange(len(my_tokens)).to(device)+1 # map zero to zero.
        self.my_importance = torch.zeros(len(my_tokens)+1).to(device)
        self.my_importance[0] = torch.finfo(torch.float).min
        self.my_importance = Parameter(self.my_importance)
        self.my_values = Parameter(torch.zeros(len(my_tokens)+1))

    def forward(self, x):
        tok_values = self.my_values[self.indexer[x]]
        tok_importance = self.my_importance[self.indexer[x]]
        #print(tok_importance)
        return torch.stack((torch.zeros(len(x), device=self.device), torch.sum(torch.softmax(tok_importance, dim=-1)*tok_values, axis=1)), dim=1)

    def _s_val_to_softmax(self, features):
        s_vals = self.my_importance[1:]
        features_pres = (features > 0).float() # 1 if some occurances. 
        feature_weights = features_pres * s_vals.reshape(1, -1)
        # account for mutliplicities. if atoken appears more then ones, e.g. n times
        # We have exp(s + log(n)) = exp(log(n)) * exp(s) = n+exp(s)
        feature_weights[features_pres>0] += torch.log(features[features_pres>0])
        feature_weights[features_pres == 0] = torch.finfo(torch.float).min
        feature_weights = torch.softmax(feature_weights, dim=1)
        return feature_weights

    def forward_feature_vects(self, x):
        alphas = self._s_val_to_softmax(x)
        v_vals = self.my_values[1:]
        output = torch.sum(alphas * v_vals.reshape(1, -1), dim=-1)
        return torch.stack((torch.zeros(len(x)).to(self.device), output.to(self.device)), dim=1)

    def set_parameters(self, v, s):
        self.my_importance.data[1:] = s
        self.my_values.data[1:] = v

    def get_importances(self):
        myrawimportance = self.my_importance[1:].detach()
        return myrawimportance - myrawimportance.mean()

    def get_values(self):
       return self.my_values[1:].detach()


def get_inputs(batch_size, vocab, seq_len, fixed_len=True, min_len=1, use_cls=True):
    start_tok = torch.tensor([101], dtype=torch.long).reshape(1, 1)
    end_tok = torch.tensor([102], dtype=torch.long).reshape(1, 1)
    if fixed_len:
        sample_len = seq_len*torch.ones(batch_size, dtype=torch.long)
    else:
        sample_len = torch.randint(seq_len-min_len+1, size=[batch_size])+min_len
    if use_cls:
        my_inputs = vocab[torch.randint(len(vocab), size=(batch_size, seq_len+1))]
        my_inputs[torch.arange(batch_size), sample_len] = end_tok
        mask = (torch.ones(batch_size, 1)*torch.arange(seq_len+1).reshape(1, -1)<=sample_len.reshape(-1,1)).long()
        inputs = torch.cat((start_tok*torch.ones(batch_size, 1), my_inputs), dim=1)
        mask = torch.cat((torch.ones(batch_size, 1), mask), dim=1)
    else:
        inputs = vocab[torch.randint(len(vocab), size=(batch_size, seq_len))]
        mask = (torch.ones(batch_size, 1)*torch.arange(seq_len).reshape(1, -1)<sample_len.reshape(-1,1)).long()
    return (inputs*mask).long(), mask.long()

def sample_dataset(batch_size, real_model, vocab, seq_len=3, use_cls=True, fixed_len=True, device="cuda"):
    inp_ids, mask = get_inputs(batch_size, vocab.cpu(), seq_len, fixed_len=fixed_len, use_cls=use_cls)
    ## Forward real model.
    with torch.no_grad():
        output = real_model(inp_ids.to(device), attention_mask=mask.to(device))["logits"]
        output = output[:,1] - output[:,0]
    output = output.detach()
    return inp_ids, mask, output

def fit_sgd_rand(example_slalom_model, real_model, vocab: torch.tensor, num_eps=10, lr=1e-3, subsize=250, batch_size=1024, 
    seq_len=3, use_cls=True, fixed_len=True, offline_ds_size = None):
    """
        OFFICIAL SLALOM fitting implementation. See ground_truth_models.SLALOMLocalExplanantions for a full implementation.
        example_slalom_model: The SLALOM Model to fit.
        real_model: The prediction model
        vocab: the vocabulary for which the SLALOM model should be fitted
        num_eps: Number of SGD epochs
        lr: The learning rate to use for SGD
        subsize: if new data batches are sampled, how many batches to use as one epoch.
        batch_size: The current batch size. Larger batch sizes appear more stable.
        seq_len: The sequence length of sequence used to fit SLALOM
        use_cls: Use a CLS token in the model
        offline_ds_size: int or None. If int, an offline dataset of offline_ds_size is sampled once and subsequently used for the fit in each epoch.
            The parameter subsize is ignored as the number of batches in one epoch will be given by offline_ds_size/batch_size
    """
    real_model = real_model.to(example_slalom_model.device)
    real_model.eval()

    my_optim = Adam(example_slalom_model.parameters(), lr = lr)
    iters = 0
    if offline_ds_size:
        inps_list, mask_list, output_list = [], [], []
        for i in range(0, offline_ds_size, batch_size):
            inps, masks, outputs = sample_dataset(batch_size, real_model, vocab, seq_len=seq_len, use_cls=use_cls, fixed_len=fixed_len, device=example_slalom_model.device)
            inps_list.append(inps)
            mask_list.append(masks)
            output_list.append(outputs)
        myds = torch.utils.data.TensorDataset(torch.cat(inps_list, dim=0), torch.cat(mask_list, dim=0), torch.cat(output_list, dim=0))
        mydl = torch.utils.data.DataLoader(myds, batch_size = batch_size)

    for ep in range(num_eps):
        losses = []
        if offline_ds_size:
            my_dl_iter = iter(mydl)
        for i in range(subsize):
            if offline_ds_size:
                try:
                    inp_ids, mask, output = next(my_dl_iter)
                except StopIteration:
                    break
            else:
                inp_ids, mask = get_inputs(batch_size, vocab.cpu(), seq_len, fixed_len=fixed_len, use_cls=use_cls)
                ## Forward real model.
                with torch.no_grad():
                    output = real_model(inp_ids.to(example_slalom_model.device), attention_mask=mask.to(example_slalom_model.device))["logits"]
                    output = output[:,1] - output[:,0]
                output = output.detach()
                #print(output.shape, output.device)
            my_optim.zero_grad()
            output_slalom = example_slalom_model.forward(inp_ids.to(example_slalom_model.device))[:,1]
            loss = torch.sum(torch.pow(output-output_slalom, 2))
            loss.backward()
            my_optim.step()
            losses.append(math.sqrt(loss.item()/len(output)))
            iters += 1
        print(sum(losses))
    return example_slalom_model.my_values[1:].detach(), example_slalom_model.my_importance[1:].detach(), example_slalom_model