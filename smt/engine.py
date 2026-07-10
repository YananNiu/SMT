import wandb
import time
import numpy as np
import torch
import torch.nn as nn
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from functools import partial
from smt.models.vit_model_collection import Layer_scale_init_Block
from smt.models.vit_model_collection import vit_model_ts, vit_model_img,vit_model_imgs,vit_model_img_ts,vit_model_2img_ts
from smt.models.cnn_model_collection import CNN_LSTM, TSImageModel
from smt.models.lstnet import LSTNet_model
from smt.train_val_test import apply_encoder

def create_encoder(args,device):
    print(f"Encoder: {args.encoder_path}")
    with open(args.encoder_path, 'rb') as f:
        encoder = torch.load(f)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder

def create_encoder_state_dict(encoder_skeleton, args,device):
    print(f"Encoder: {args.encoder_path}")
    state_dict = torch.load(args.encoder_path)
    encoder_skeleton.load_state_dict(state_dict)
    encoder_skeleton.to(device)
    encoder_skeleton.eval()
    return encoder_skeleton

def create_model(args,img_size,ts_size,in_chans,img_num):
    if args.model_skeleton == 'vit_model_ts':
        model = eval(f"""{args.model_skeleton}(ts_shape = ts_size,
                     embed_dim=args.embed_dim, depth=args.depth_transformer, num_heads=args.num_heads, mlp_ratio=4, qkv_bias=True,
                     qk_scale=None, attn_drop_rate=args.attn_drop,drop_rate=args.drop_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block)""")
    elif args.model_skeleton == 'vit_model_img_ts' or args.model_skeleton == 'vit_model_2img_ts':
        model = eval(f"""{args.model_skeleton}(img_size=img_size,  patch_size=16, in_chans=in_chans, 
                     ts_shape = ts_size,
                     embed_dim=args.embed_dim, depth=args.depth_transformer, num_heads=args.num_heads, mlp_ratio=4, qkv_bias=True,
                     qk_scale=None, attn_drop_rate=args.attn_drop,drop_rate=args.drop_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, patch_mode=args.patch_mode)""")
    elif args.model_skeleton == 'vit_model_img':
        model = eval(f"""{args.model_skeleton}(img_size =img_size,patch_size=16,in_chans=in_chans, 
                     embed_dim=args.embed_dim, depth=args.depth_transformer, num_heads=args.num_heads, mlp_ratio=4, qkv_bias=True,
                     attn_drop_rate=args.attn_drop,drop_rate=args.drop_rate,norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, patch_mode=args.patch_mode)""")
    elif args.model_skeleton == 'vit_model_imgs':
        model = eval(f"""{args.model_skeleton}(img_size =img_size,patch_size=16,in_chans=in_chans,
                        embed_dim=args.embed_dim, depth=args.depth_transformer, num_heads=args.num_heads, mlp_ratio=4, qkv_bias=True,
                        img_num = img_num,
                        attn_drop_rate=args.attn_drop,drop_rate=args.drop_rate,norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, patch_mode=args.patch_mode)""")
    elif args.model_skeleton == 'CNN_LSTM':
        model = eval(f"""{args.model_skeleton}(input_shape=(args.batch_size,in_chans,img_size[0],img_size[1]))""")
    elif args.model_skeleton == 'CNNLSTM_2camera':
        model = CNN_LSTM(input_shape=(args.batch_size,2,in_chans,img_size[0],img_size[1]))
    elif args.model_skeleton == 'CNNLSTM_LSTNet':
        model_ts = LSTNet_model(args,num_factors=ts_size[0])
        model_img = CNN_LSTM(input_shape=(args.batch_size,in_chans,img_size[0],img_size[1]))
        model = TSImageModel(model_img, model_ts)
    elif args.model_skeleton == 'LSTNet':
        model = LSTNet_model(args, num_factors=ts_size[0])
    return model

def make_data_img(Datagenerator, args,img_num):
    if args.model_skeleton == 'CNNLSTM_2camera':
        # two cameras stacked into (B, 2, C, H, W); no time series
        kw = dict(image1=args.image1, image_time1=args.image_time1,
                  image2=args.image2, image_time2=args.image_time2,
                  ts_data=args.ts_data, horizon=args.horizon, window=args.window,
                  data_flag=args.data_flag, indices=args.indices,
                  special_test=args.special_test, creat_real_test=False,
                  image_token=True, ts_token=False, image_stack=True)
        train_data = Datagenerator(flag='train', **kw)
        valid_data = Datagenerator(flag='val', **kw)
        test_data = Datagenerator(flag='test', **kw)
        return train_data, valid_data, test_data
    if img_num != None:
        train_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, 
                                   flag = 'train',img_num=args.img_num,data_flag=args.data_flag,indices=args.indices,creat_real_test=False,
                                   image_token = args.image_token)
        valid_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, 
                                flag = 'val',img_num=args.img_num, data_flag=args.data_flag,indices=args.indices,creat_real_test=False,
                                image_token = args.image_token)
        test_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, 
                                flag = 'test',img_num=args.img_num, data_flag=args.data_flag,indices=args.indices,creat_real_test=False,
                                image_token = args.image_token)
    elif args.ts_token!=None:
        train_data = Datagenerator(image1 = args.image1, image_time1 = args.image_time1,image2 = args.image2, image_time2 = args.image_time2,
                                   ts_data=args.ts_data,horizon=args.horizon, window=1,
                                   flag = 'train',data_flag=args.data_flag,indices=args.indices,creat_real_test=False,
                                   image_token = args.image_token,ts_token=args.ts_token)
        valid_data = Datagenerator(image1 = args.image1, image_time1 = args.image_time1,image2 = args.image2, image_time2 = args.image_time2,
                                   ts_data=args.ts_data,horizon=args.horizon, window=1,
                                flag = 'val',data_flag=args.data_flag,indices=args.indices,creat_real_test=False,
                                image_token = args.image_token,ts_token=args.ts_token)
        test_data = Datagenerator(image1 = args.image1, image_time1 = args.image_time1,image2 = args.image2, image_time2 = args.image_time2,
                                  ts_data=args.ts_data,horizon=args.horizon, window=1,
                                flag = 'test',data_flag=args.data_flag,indices=args.indices,creat_real_test=False,
                                image_token = args.image_token,ts_token=args.ts_token)
    else:    
        train_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                flag = 'train',data_flag=args.data_flag,indices=args.indices,creat_real_test=False,special_test=args.special_test,
                                image_token=args.image_token,ts_token=args.ts_token)
        valid_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                flag = 'val',data_flag=args.data_flag,indices=args.indices,creat_real_test=False,special_test=args.special_test,
                                image_token=args.image_token,ts_token=args.ts_token)
        test_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                flag = 'test',data_flag=args.data_flag,indices=args.indices,creat_real_test=False,special_test=args.special_test,
                                image_token=args.image_token,ts_token=args.ts_token)

    return train_data, valid_data, test_data

def make_data_vilt(Datagenerator, args):
    if args.model_skeleton == 'vit_model_2img_ts': # use Datagenerator_ViLT_imgs
        train_data = Datagenerator(image1 = args.image1, image_time1 = args.image_time1,image2 = args.image2, image_time2 = args.image_time2,
                                   ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                   flag = 'train',meteo=args.meteo,data_flag=args.data_flag,indices=args.indices,
                                   image_token = args.image_token,creat_real_test=False)
        valid_data = Datagenerator(image1 = args.image1, image_time1 = args.image_time1,image2 = args.image2, image_time2 = args.image_time2,
                                   ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                   flag = 'val',meteo=args.meteo,data_flag=args.data_flag,indices=args.indices,
                                   image_token = args.image_token,creat_real_test=False)
        test_data = Datagenerator(image1 = args.image1, image_time1 = args.image_time1,image2 = args.image2, image_time2 = args.image_time2,
                                   ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                   flag = 'test',meteo=args.meteo,data_flag=args.data_flag,indices=args.indices,
                                   image_token = args.image_token,creat_real_test=False)
    else: # use Datagenerator_ViLT 
        train_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                    flag = 'train',meteo=args.meteo,data_flag=args.data_flag,indices=args.indices,
                                    image_token = args.image_token,ts_token=args.ts_token,creat_real_test=False,special_test=args.special_test)
        valid_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                flag = 'val',meteo=args.meteo,data_flag=args.data_flag,indices=args.indices,
                                image_token = args.image_token,ts_token=args.ts_token,creat_real_test=False,special_test=args.special_test)
        test_data = Datagenerator(image = args.image, image_time = args.image_time,ts_data=args.ts_data,horizon=args.horizon, window=args.window,
                                flag = 'test',meteo=args.meteo,data_flag=args.data_flag,indices=args.indices,
                                image_token = args.image_token,ts_token=args.ts_token,creat_real_test=False,special_test=args.special_test)
    return train_data, valid_data, test_data



def make_vilt(Datagenerator, args, device,**kwargs):
    assert args.model_skeleton in ['vit_model_ts','vit_model_img','vit_model_imgs','vit_model_img_ts','vit_model_2img_ts',
                                   'CNN_LSTM','CNNLSTM_2camera','CNNLSTM_LSTNet','LSTNet']
    
    img_num = kwargs.get('img_num')

    #make the data
    if args.model_skeleton in ['vit_model_img','CNNLSTM_2camera','vit_model_imgs']:
        train_data, valid_data, test_data = make_data_img(Datagenerator, args, img_num)
    else:
        train_data, valid_data, test_data = make_data_vilt(Datagenerator, args)
    
    # get data dim
    encoder = kwargs.get('encoder')
    if encoder:
        x = torch.from_numpy(train_data.pixel_values[np.atleast_1d(0)]).float().to(device)
        feature = apply_encoder(x,encoder) 
        img_size=(feature.shape[2],feature.shape[3])
        in_chans = feature.shape[-3]
    else:
        img_size=(test_data.pixel_values.shape[2],test_data.pixel_values.shape[3])
        in_chans = test_data.pixel_values.shape[-3]
        
    if args.model_skeleton == 'vit_model_img' or args.model_skeleton == 'CNN_LSTM':
        ts_size = None
    else:
        ts_size = (test_data.rawdat.shape[1], test_data.rawdat.shape[2])

    # make the model
    model = create_model(args,img_size,ts_size,in_chans,img_num)
    pretrained_model = kwargs.get('pretrained_model')
    if pretrained_model!=None:
        state_dict = torch.load(pretrained_model)
        model.load_state_dict(state_dict)
        print(f"Pretrained model loaded from {pretrained_model}")
    else:
        print("Train from scratch")
    model.to(device)
    
    # create criterion
    criterion = nn.MSELoss(reduction='sum')
    evaluateL2 = nn.MSELoss(reduction='sum')
    best_val = 10000000

    # create optimizer and lr_scheduler
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    return model, train_data, valid_data, test_data, criterion, evaluateL2, optimizer, lr_scheduler, best_val



def run_model(model, train,evaluate,train_data, valid_data, test_data, criterion, evaluateL2, optim, lr_scheduler, best_val,args,model_save_path,device,bootstrap_idx=[],**kwargs):
    encoder = kwargs.get('encoder')
    try:                    
        epochs_without_improvement = 0
        print('begin training', args.epochs_max)
        for epoch in range(1, args.epochs_max+1):
            epoch_start_time = time.time()
            train_rmse,lr =train(train_data, model= model, criterion=criterion, optim=optim, batch_size = args.batch_size,device=device,bootstrap_idx = bootstrap_idx,encoder=encoder)
            wandb.log({ "Train rmse": train_rmse, "lr": lr})
            #if lr_scheduler is not None:
            lr_scheduler.step(epoch)
            val_rmse,val_rse,val_corr = evaluate(valid_data, model = model, evaluateL2=evaluateL2, batch_size = args.batch_size,device=device,encoder=encoder)
            wandb.log({ "Val rmse": val_rmse,"Val rse": val_rse,"Val corr": val_corr})
            print('| end of epoch {:3d} | time: {:5.2f}s | train_rmse {:5.4f} | valid rmse {:5.4f} | valid rse {:5.4f} |  valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_rmse, val_rmse, val_rse, val_corr))

            # Save the model if the validation loss is the best we've seen so far.
            if val_rmse < best_val:
                with open(model_save_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_val = val_rmse
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print("Early stopping, no improvement for %d epochs with best validation rmse %5.4f" % (args.patience, best_val))
                    break
            wandb.log({ "best_val_loss": best_val})
            if epoch % 5 == 0:
                test_rmse,test_rse,test_corr  = evaluate(test_data, model = model, evaluateL2=evaluateL2, batch_size = args.batch_size,device=device,encoder=encoder)
                print ("test rmse {:5.4f} |test rse {:5.4f} | test corr {:5.4f}".format(test_rmse,test_rse,test_corr))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        
