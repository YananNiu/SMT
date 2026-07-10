import torch
import math
import numpy as np
import pandas as pd

def evaluate_oneEpoch(data, model, evaluateL2, batch_size,device,**kwargs):
    model.eval()
    total_loss = 0
    n_samples = 0
    predict = None
    test = None
    encoder = kwargs.get('encoder')
    with torch.no_grad():
        for batch in data.get_batches(batch_size=batch_size, device=device, shuffle=False):
            batch_list = list(batch)
            T = batch_list.pop()
            Y = batch_list.pop()
            # evaluate every sample: metrics are sum-reduced and divided by the
            # true sample count, so partial (non batch_size) batches are fine.

            if not encoder:
                pass
            else:
                encoded_first_element = apply_encoder(batch_list.pop(0),encoder)
                batch_list.insert(0, encoded_first_element)
            
            output = model(*batch_list).squeeze(1)
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict,output))
                test = torch.cat((test, Y))

            total_loss += evaluateL2(output, Y ).item()
            n_samples += output.size(0)

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()

        rmse = math.sqrt(total_loss / n_samples)
        rse = rmse/Ytest.std(ddof=1)
        correlation = ((predict - predict.mean()) * (Ytest - Ytest.mean())).mean()/(predict.std() * Ytest.std())
        # same as np.corrcoef(predict, Ytest)[0, 1]
        
    return rmse, rse, correlation

def train_oneEpoch(data, model, criterion, optim,batch_size,device,bootstrap_idx=False,**kwargs):
    model.train()
    total_loss = 0
    n_samples = 0
    encoder = kwargs.get('encoder')
    for batch in data.get_batches(batch_size=batch_size, device=device, shuffle=True,bootstrap_idx=bootstrap_idx):
        batch_list = list(batch)
        T = batch_list.pop()
        Y = batch_list.pop()
        if not encoder:
            pass
        else:
            encoded_first_element = apply_encoder(batch_list.pop(0),encoder)
            batch_list.insert(0, encoded_first_element)
        
        output = model(*batch_list).squeeze(1)
        loss = criterion(output,Y)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        n_samples += output.size(0)

    return math.sqrt(total_loss / n_samples), optim.param_groups[0]['lr']

def test_oneEpoch(data, model, evaluateL2, batch_size,device,**kwargs):
    model.eval()
    total_loss = 0
    n_samples = 0
    predict = None
    test = None
    time = None
    encoder = kwargs.get('encoder')
    with torch.no_grad():
        for batch in data.get_batches(batch_size=batch_size, device=device, shuffle=False):
            batch_list = list(batch)
            T = batch_list.pop()
            Y = batch_list.pop()
            
            if not encoder:
                pass
            else:
                encoded_first_element = apply_encoder(batch_list.pop(0),encoder)
                batch_list.insert(0, encoded_first_element)
            
            output = model(*batch_list).squeeze(1)
            if predict is None:
                predict = output
                test = Y
                time = T
            else:
                predict = torch.cat((predict,output))
                test = torch.cat((test, Y))
                time = np.concatenate((time,T))

            total_loss += evaluateL2(output, Y ).item()
            n_samples += output.size(0)

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()

        rmse = math.sqrt(total_loss / n_samples)
        rse = rmse/Ytest.std()
        correlation = ((predict - predict.mean()) * (Ytest - Ytest.mean())).mean()/(predict.std() * Ytest.std())
    #print ("test rmse {:5.4f} | test rse {:5.4f}| test corr {:5.4f}".format(rmse,rse,correlation))
    df = pd.DataFrame({'time':time,'predict':predict,'test':Ytest})
    # predictions are h steps ahead; shift timestamps forward by the horizon if needed
    #df['time'] = df['time']+pd.Timedelta(hours=2)
    return df


def apply_encoder(x,encoder):
    encoder.eval()
    with torch.no_grad():
        feature= encoder(x, return_features=True)
        feature = feature.reshape(feature.shape[0],1,feature.shape[1],feature.shape[2]*feature.shape[3])
    return feature