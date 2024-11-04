# ### Implement ACE and ConceptSHAP

# import  skimage.segmentation as segmentation
# #import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image, ImageFont, ImageDraw 
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
# from multiprocessing import dummy as multiprocessing
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import time
import torch.optim as optim
# import matplotlib.cm as cmap
import os

################################ CONCEPTSHAP IMPLEMENTATION ######################################

class ConceptLatentMapping(nn.Module):
    """ The reconstruction of the latent activation given concept vectors. """
    def __init__(self, num_latent_vec = 49, n_concept_vectors = 50, num_channels = 2048, n_interm = 50):
        super(ConceptLatentMapping, self).__init__()
        self.num_latent_vec = num_latent_vec # m
        self.n_concept_vectors = n_concept_vectors # T
        self.num_channels = num_channels # d
        self.linear1 = nn.Linear(self.num_latent_vec*self.n_concept_vectors, n_interm)
        self.linear2 = nn.Linear(n_interm, self.num_channels)
        
    def forward(self, x):
        """ Predict the latent space out of the concept vectors 
            input shape: (batch, num_latent_vec, n_concept_vectors)
            output shape (batch, num_latent_vec, num_channels)
        """
        x = torch.sigmoid(self.linear1(x.view(x.size(0),-1)))
        return self.linear2(x).view(x.size(0), -1)

class ConceptBERT(nn.Module):
    def __init__(self, base_model, n_concept_vectors = 50, num_classes = 2, snippet_len=10, force_orthogonal=True):
        """ Init the Completeness Aware-Concept Model from Yeh et al. (2020)
            base_model: src.classification_models BERT or RoBERTa model (tested with pretrained)
            n_concept_vectors: Number of concept vectors to use,
            num_classes: Number of classes
            use_layer: which hidden states to use: -1 = last layer.
        """
        super(ConceptBERT, self).__init__()
        #print(self.__dict__)
        self.n_concept_vectors = n_concept_vectors
        self.force_orth = force_orthogonal
        self.use_layer = -1
        self.base_model = base_model.model.model
        self.base_model.eval()
        self.tokenizer = base_model.tokenizer
        self.n_channels = self.base_model.config.hidden_size
        self.n_interm = 100
        self.snippet_len = snippet_len
        self.num_snippets = self.tokenizer.model_max_length // self.snippet_len
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        for p in self.base_model.parameters():
            p.requires_grad_(False)
    
        self.conceptVectors = Parameter(torch.randn(self.n_channels , self.n_concept_vectors)) # Non orthogonal, non normalized.
        # pixel_sides = [input_size//4, input_size//4, input_size//8, input_size//16, input_size//32]
        # self.n_latentvectors = pixel_sides[self.latent_level]*pixel_sides[self.latent_level]
        self.latentMapping = ConceptLatentMapping(self.num_snippets, self.n_concept_vectors, self.n_channels, self.n_interm)
        

    def obtain_subpatches(self, input_ids, attention_mask, usecls=True):
        """ Split the sequence in valid subpatches of a common length self.snippet_len tokens. """
        input_ids_snip = input_ids[:, 1:self.num_snippets*self.snippet_len+1] # Cut off CLS
        attention_mask_snip = attention_mask[:, 1:self.num_snippets*self.snippet_len+1]
        attention_mask_snip = attention_mask_snip.reshape(-1, self.snippet_len)
        input_ids_snip = input_ids_snip.reshape(-1, self.snippet_len)
        last_invalid = torch.logical_or((input_ids_snip[:,-1] == self.eos_token_id), (input_ids_snip[:,-1] == self.pad_token_id)).long() # is the last token invalid?
        first_invalid = torch.logical_or((input_ids_snip[:,0] == self.eos_token_id), (input_ids_snip[:,0] == self.pad_token_id)).long() # is the first token invalid?
        input_ids_snip  = torch.cat(((1-first_invalid).reshape(-1,1)*self.cls_token_id+first_invalid.reshape(-1,1)*self.pad_token_id,
                input_ids_snip, (1-last_invalid.reshape(-1,1))*self.eos_token_id+ last_invalid.reshape(-1,1)*self.pad_token_id), dim=1)

        attention_mask_snip = torch.cat(((1-first_invalid).reshape(-1,1), attention_mask_snip, (1-last_invalid).reshape(-1,1)), dim=1)
        return input_ids_snip, attention_mask_snip


    def forward_bert(self, input_ids, attention_mask):
        """ Implementation of the forward for the resnet, that additionally returns the latent vectors.
            Copied from https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
            x: Batch of images 
            Returns: prediction, latent variables selected by the model
        """
        output = self.base_model.forward(input_ids, attention_mask=attention_mask)
        x = output["logits"]
        ## Reshape the model inputs to patches and forward them.
        input_ids_snip, attention_mask_snip = self.obtain_subpatches(input_ids, attention_mask)
        output_latent = self.base_model.forward(input_ids_snip, attention_mask=attention_mask_snip, output_hidden_states=True)
        latents = output_latent["hidden_states"][self.use_layer][:, 0].reshape(len(x), self.num_snippets, -1)
        return x, latents

    def forward_bert_mapping(self, input_ids, attention_mask):
        """
            Forward the Bert model AND the latentConceptMapping to obtain both predictions
            Returns: predictions of the resnet, precictions by concept scores, latent vectors 
        """
        x, latents = self.forward_bert(input_ids, attention_mask)
        scores = self.compute_concept_scores(latents, attention_mask)
        z = self.latentMapping(scores).view(latents.size(0), latents.size(2))

        ## Bert classification head wrapper
        z = self.classification_head(z)
        return x, z, latents

    def classification_head(self, inputs: torch.Tensor):
        if "bert" in self.base_model.__dict__: ## Bert based
            x = self.base_model.bert.pooler.dense(inputs)
            x = self.base_model.bert.pooler.activation(x)
            return self.base_model.classifier(x) 
        else:
            return self.base_model.classifier(inputs.unsqueeze(1))
        #pooled_output = self.base_model.pre_classifier(inputs)  # (bs, dim)
        #pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        #pooled_output = model_trained.dropout(pooled_output)  # (bs, dim)
         # (bs, num_labels)

    def normal_cv(self):
        """ get normalized concept vectors. If force_orthogonal, they are constrained to be orthogonal.
            Shape(n_channels, n_concept_vectors)
        """
        if self.force_orth:
            mean_norm = self.conceptVectors - self.conceptVectors.mean(dim=0, keepdim=True) # Substract vector mean
            eig_val, D = torch.linalg.eigh(mean_norm.transpose(0,1).matmul(mean_norm))
            return D.matmul(torch.diag(torch.pow(eig_val, -0.5)).matmul(D.transpose(0,1).matmul(mean_norm.transpose(0,1)))).transpose(0,1) #D*Lambda^(-0.5)*D^T * mean_norm
        else:
            return self.conceptVectors / torch.norm(self.conceptVectors, p=2, dim=0, keepdim=True)

    def forward(self, x):
        """ Forward pass of the model. """
        output, z, latents = self.forward_bert_mapping(x)
        return output, z, latents

    def compute_concept_scores(self, layer5latents, attention_mask):
        """ Compute the normalized concept scores.
            layer5latents: Latent variables of Shape (batch, n_channels, hidden_size) 
            attention_mask: rule out concepts with no valid tokens.
            Return concept scores (batch, n_concept_vectors)
        """
        dot_prods = torch.matmul(layer5latents,
                                self.normal_cv()) # shape(batch, n_snippets, n_concept_vectors)

        # Normalize over all concepts
        dot_prods = torch.relu(dot_prods)
        dot_prods_n = dot_prods / (torch.norm(dot_prods, p=2, dim=2, keepdim=True)+0.01)
        ## Rule out invalid
        attention_mask_snip = attention_mask[:, 1:self.num_snippets*self.snippet_len+1]
        attention_mask_snip = attention_mask_snip.reshape(-1, self.num_snippets, self.snippet_len).sum(-1)
        return dot_prods_n*(attention_mask_snip.unsqueeze(-1) > 1)
    
    def surr_completeness(self, z, y):
        """ Surrogate completeness loss (eqn. 3) """
        return nn.CrossEntropyLoss().__call__(z, y)

    def divergence_loss(self):
        """ sum c_i * c_j - diagonal elements. Penalize vectors that are not ortogonal."""
        if self.force_orth:
            return 0.0
        #print(torch.sum(self.normal_cv().transpose(0,1).matmul(self.normal_cv()).flatten()), torch.sum(torch.pow(self.normal_cv(), 2.0).flatten()))
        return (torch.sum(self.normal_cv().transpose(0,1).matmul(self.normal_cv()).flatten()) - torch.sum(torch.pow(self.normal_cv(), 2.0).flatten())) / \
    ((self.n_concept_vectors)*(self.n_concept_vectors-1))
        
    def nearest_neighbor_loss(self, layer5latents, K=3):
        """ Compute nearest neighbors to concepts """
        # shape (b, 51, 768)
        lat_flat = layer5latents.transpose(0, 2)
        lat_flat= lat_flat.reshape(lat_flat.size(0), -1) # shape (768, 2*51)
        dmat = torch.matmul(self.normal_cv().transpose(0,1), lat_flat)
        # (n_concepts, x)
        # find K-nearest neighbors.
        nearest = torch.topk(dmat, k=K, dim=1, largest=True, sorted=False)[0]
        #print(nearest.shape, torch.sum(nearest.flatten()).item())
        return torch.sum(nearest.flatten())/(K*self.n_concept_vectors)




def find_concepts(mtrain_loader, model, N=15, device="cuda", len_tokens=10):
    """ Maintain a list of the N closest activations for each concept identified. 
        mtrain_loader: The train loader
        model: ConceptResNet
        N: Number of closest samples to list+
        use_gpu: CUDA id to use, -1 for cpu.
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    #losses = AverageMeter('Loss', ':.4e')
    #top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(mtrain_loader),
        [batch_time, data_time],
        prefix="Batch:")

    udevice = device
    
    # switch to train mode
    model.eval()
    model = model.to(udevice)
    end = time.time()
    with torch.no_grad():
        concept_vectors = model.normal_cv()
        closest_class = torch.zeros(concept_vectors.size(1), N, dtype=torch.long, device=udevice)
        closest_dots = torch.zeros(concept_vectors.size(1), N, device=udevice)
        closest_imgids = torch.zeros(concept_vectors.size(1), N, device=udevice, dtype=torch.long)
        closest_tokens = torch.zeros(concept_vectors.size(1), N, len_tokens, device=udevice, dtype=torch.long)
        batch_size = mtrain_loader.batch_size

        indexoffsets =  N*torch.arange(0, concept_vectors.size(1), device=udevice).reshape(-1,1).repeat(1, N) # my index offsets. one row of 0, N, 2N, 3N, ...
        for i, data in enumerate(mtrain_loader):
            images, attn_mask, labels = data["input_ids"], data["attention_mask"], data["label"]
            sample_ids = torch.arange(batch_size*i, batch_size*i+len(images), device=udevice)
            # measure data loading time
            images = images.to(udevice)
            attn_mask = attn_mask.to(udevice)
            labels = labels.to(udevice)
            # labels = labels.cuda(use_gpu, non_blocking=True)
            data_time.update(time.time() - end)
            # compute output
            _, latents = model.forward_bert(images, attn_mask)
            input_ids_snip = images[:, 1:model.num_snippets*model.snippet_len+1]
            input_ids_snip = input_ids_snip.reshape(-1, model.num_snippets, model.snippet_len)
            attnm_snip = attn_mask[:, 1:model.num_snippets*model.snippet_len+1]
            attnm_snip = attnm_snip.reshape(-1, model.num_snippets, model.snippet_len)
            #print(input_ids_snip.shape)
            # compute dot product
            dot_prods = torch.matmul(latents, model.normal_cv()) # shape(batch, n_snippets, n_concept_vectors)
            dot_prods = torch.relu(dot_prods)
            dot_prods = dot_prods / (torch.norm(dot_prods, p=2, dim=2, keepdim=True)+0.01)
            valid = attnm_snip.sum(-1) > 1
            dot_prods = dot_prods*valid.unsqueeze(-1)
            #print(dot_prods.shape)
            dot_prods_max, patch_id = torch.max(dot_prods, dim=1) # Shape (batch, n_concept_vectors)
            merged_dots = torch.cat((closest_dots, dot_prods_max.transpose(0,1)), dim=1) # Merged version have shape (n_concept_vectors, batch)
            #merged_patch = torch.cat((closest_patches, patch_id.transpose(0,1)), dim=1)

            merged_class = torch.cat((closest_class, labels.repeat(concept_vectors.size(1),1)), dim=1)
            merged_ids = torch.cat((closest_imgids, sample_ids.repeat(concept_vectors.size(1),1)), dim=1)
            max_token_sequence = torch.gather(input_ids_snip, 1, patch_id.unsqueeze(-1).repeat(1, 1, len_tokens)).transpose(0, 1)  #The sequences with the highest activations.
            #print(max_token_sequence.shape, patch_id.shape) # max token seq (10, 2, 10)
            merged_tokens = torch.cat((closest_tokens, max_token_sequence), dim=1) #images.unsqueeze(0).repeat(concept_vectors.size(1), 1, 1)), dim=1)
            #print(merged_tokens.shape)
            # Find closest vectors
            closest_dots, ind_closest = torch.topk(merged_dots, N, dim=1)
            
            # Update patches
            #closest_patches = torch.gather(merged_patch, 1, ind_closest)
            closest_class = torch.gather(merged_class, 1, ind_closest)
            closest_imgids = torch.gather(merged_ids, 1, ind_closest)
            #print(ind_closest)
            closest_tokens = torch.gather(merged_tokens, 1, ind_closest.unsqueeze(-1).repeat(1, 1, len_tokens))
            #print(closest_tokens.shape)
            # Update images
            #merged_img = torch.cat((closest_img.view(-1, closest_img.size(2), closest_img.size(3), closest_img.size(4)), images.cpu()), dim=0) # N*n_concept_vectors + batchsize
            #print(merged_img.shape)

            batch_time.update(time.time() - end)
            if i % 100 == 0:
                progress.display(i)
                #if i > 0:
                #    return closest_dots, saliency_maps, closest_class, closest_imgids
            end = time.time()

    return closest_dots, closest_class, closest_imgids, closest_tokens


def export_concepts(closest_patches, closest_img, closest_class, class_names, filename, llev=4, input_size=448, res=200):
    """ Export the concepts as PNG image """
    # Write the file to output concepts.png
    n_examples = closest_img.size(1)
    n_concept = closest_img.size(0)
    print(n_examples, n_concept)
    closest_img = closest_img.reshape(-1, 3, input_size, input_size)
    closest_patches = closest_patches.reshape(-1, input_size//32, input_size//32)
    num_idx = [56, 56, 28, 14, 7]
    mod_img = modify_plot_frames(closest_img, closest_patches, closest_class.flatten(), flen=112,
                                num_column_idx=input_size//32, num_row_idx=input_size//32, labels=class_names)
    # Downsample the images.


    img = make_grid(mod_img, nrow=n_examples, padding=2)
    npimg = img.transpose(0,1).transpose(1,2).numpy()
    print(npimg.shape)
    pil_im = Image.fromarray(np.uint8(npimg*255)).resize((n_examples*res, n_concept*res), Image.BILINEAR)
    pil_im.save(filename)


def validate_concept(val_loader, model, criterion, use_gpu, print_freq, multiple_out = True):
    """ Validate the normal prediction of the resnet on the hold out testset (e.g. to check if the pretrained network was correctly loaded.)
        val_loader: Test loader
        model: Model (normal resnet, or ConceptResNet), set multiple_out to false for plain resnet.
        criterion: Loss fn
        use_gpu: Id of gpu to use
        print_freg: Printout frequency. Will print to the Jupyter-Server command line! (this allows to see outputs if the script continues running with the browser closed.)
        multiple_out: True, if the model returns multiple outputs, with the first being the prediction. False, if it only returns the prediction.
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images = data[0].cuda(use_gpu, non_blocking=True)
            target = data[1].cuda(use_gpu, non_blocking=True)

            # compute output
            #if multiple_out:
            #    loss, output = criterion(model, images, target)
            #else:
            output = model(images)
            loss = criterion(output[0], target)
            

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output[0], target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg.item(), top5.avg.item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        #print('\t'.join(entries))
        os.write(1, ('\t'.join(entries)+"\n").encode())

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

def concept_loss(model, x, attn, y, l1 = 2.0, l2=1.0):
    pred, z, latents = model.forward_bert_mapping(x, attn)
    a = model.surr_completeness(z, y)
    b = model.nearest_neighbor_loss(latents, K=3)
    c = model.divergence_loss()
    #print([el.detach().cpu().item() for el in [a,b,c]])
    loss = a - l1*b + l2*c
    return loss, z

class ConceptSHAP:
    """ Our implementation of ConceptSHAP (discovery stage only) """

    def __init__(self, baseresnet, n_concepts=50, num_classes=200, force_orth = True, snippet_len=10):
        self.n_concepts = n_concepts
        self.num_classes = num_classes
        self.force_orth = force_orth
        self.snippet_len = snippet_len
        self.cresnet = ConceptBERT(baseresnet, n_concept_vectors = self.n_concepts, num_classes = self.num_classes,
            snippet_len=self.snippet_len, force_orthogonal=self.force_orth)

    def discover_concepts(self, train_data_loader, device="cuda", n_epochs=3):
        optimizer = optim.Adam(self.cresnet.parameters(), lr=1e-3)
        self.cresnet.train()
        self.cresnet = self.cresnet.to(device)
        for epoch in range(0, n_epochs):
            print("Epoch", epoch)
            if epoch == 1:
                optimizer = optim.Adam(self.cresnet.parameters(), lr=5e-4)
            if epoch >= 2:
                optimizer = optim.Adam(self.cresnet.parameters(), lr=1e-4)
            self._train_concept(train_data_loader, self.cresnet, concept_loss, optimizer, epoch, device=device)

    def _train_concept(self, mtrain_loader, model, criterion, optimizer, epoch, device="cuda"):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(mtrain_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        for i, data in enumerate(mtrain_loader):
            images, attn_mask, target = data["input_ids"], data["attention_mask"], data["label"]
            # measure data loading time
            images = images.to(device)
            attn_mask = attn_mask.to(device)
            target = target.to(device)
            #print(images.dtype, images.shape)
            #print(target.dtype, target.shape)
            #print("On GPU : ", args.gpu, images.shape, target.shape)
            data_time.update(time.time() - end)
            # compute output
            loss, z = criterion(model, images, attn_mask, target)
            #print(loss, z)
            # measure accuracy and record loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc1 = accuracy(z.detach(), target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            # measure elapsed time  
            batch_time.update(time.time() - end)
            if i % 20 == 0:
                progress.display(i)
            end = time.time()
