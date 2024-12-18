import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, wandb=None, tokenizer=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, ids, query_caption, reference_caption in bar:
        
        if scaler:
            with autocast():
                query_caption = tokenizer(query_caption, padding=True, truncation=True, return_tensors="pt")
                reference_caption = tokenizer(reference_caption, padding=True, truncation=True, return_tensors="pt")
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                query_caption = {k: v.to(train_config.device) for k, v in query_caption.items()}
                reference_caption = {k: v.to(train_config.device) for k, v in reference_caption.items()}
            
                # Forward pass
                features1, features2, querry_text_features, reference_text_features = model(query, reference, query_caption, reference_caption)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, querry_text_features, reference_text_features, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, querry_text_features, reference_text_features, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            query_caption = tokenizer(query_caption, padding=True, truncation=True, return_tensors="pt")
            reference_caption = tokenizer(reference_caption, padding=True, truncation=True, return_tensors="pt")
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            
            # Forward pass
            features1, features2, querry_text_features, reference_text_features = model(query, reference, query_caption, reference_caption)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, querry_text_features, reference_text_features, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, querry_text_features, reference_text_features, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            if wandb:
                wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader, is_query=True):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                img = img.to(train_config.device)
                if is_query:
                    img_feature = model(img)
                else:
                    img_feature = model(None, img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list