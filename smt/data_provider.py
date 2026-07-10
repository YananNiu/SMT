import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from skimage import exposure
import random
            
class DataGenerator_ViLT(object):
    def __init__(self, image, image_time,ts_data,horizon, window,flag,meteo=False,data_flag='GHI_percent_wrt_max',
                 indices='auto',image_token = False,ts_token=False, smart_token=False,creat_real_test=False,align_image=True,special_test=False):
        # what data to use
        self.image_token = image_token
        self.ts_token = ts_token
        self.smart_token = smart_token
        self.align_image = align_image
        # img data
        self.pixel_values = np.load(image, mmap_mode='c', allow_pickle=True)
        if self.pixel_values.shape[-1] == 3 or self.pixel_values.shape[-1]==1:
            self.pixel_values = np.transpose(self.pixel_values,(0,3,1,2))
        self.image_times = np.load(image_time,allow_pickle=True).astype('datetime64[s]')
        
        # ts data
        self.P = window
        self.h = horizon
        self.flag = flag
        assert self.flag in ['train', 'val', 'test','all_test']
        self.creat_real_test = creat_real_test
        self.meteo_flag = meteo
        if meteo == True:
            self.meteo = [data_flag,'ghi_clear_sky','Relative_air_humidity', 'Air_temperature','Wind_speed', 'Wind_direction', 'Gust_peak']
            #self.meteo = ['Relative_air_humidity', 'Air_temperature','Wind_speed', 'Wind_direction', 'Gust_peak']
            #self.meteo.append(data_flag)
        else:
            self.meteo = [data_flag]
            #self.meteo = [data_flag,'ghi_clear_sky']
        self.data_flag = data_flag
        self._get_sequence(ts_data)
 
        #remove nan from ts data, from labels, and get zero clear sky index
        self.index = self._remove_nan()
        self.index_zeroClearSky = self._zeroClearSky()
        self.len = len(self.index)  

        # indices can be None or a dict of tuples (start, end) at length of 2
        #assert indices == 'auto' or (isinstance(indices, dict) and len(indices) == 3)  
        self.special_test = special_test
        if self.special_test:
            self._split_index_special(indices)
        else:
            self._split_index(indices)

        if self.rawdat.ndim == 2:
            self.rawdat = np.expand_dims(self.rawdat, axis=2)
        self.rawdat = np.transpose(self.rawdat, (0, 2, 1))
        
    def _get_sequence(self,ts_data):
        self.df = pd.read_csv(ts_data)
        self.df['time'] = pd.to_datetime(self.df['time']).values.astype('datetime64[s]')
        
        n = self.df.shape[0]-self.h-self.P+1
        self.times = np.zeros((n),dtype='datetime64[s]')
        self.rawdat = np.zeros((n,self.P,len(self.meteo)))
        self.labels = np.zeros((n))
        self.smart_index = np.zeros((n))
        for i in range(n):
            end_idx = i+self.P-1
            self.times[i] = self.df.loc[end_idx, 'time']
            self.rawdat[i] = self.df.loc[end_idx-self.P+1:end_idx, self.meteo].values
            
            self.labels[i] = self.df.loc[end_idx+self.h, self.data_flag]
            # add smart index
            if np.isnan(self.df.loc[end_idx, 'ghi_clear_sky']) or self.df.loc[end_idx, 'ghi_clear_sky'] == 0:
                self.smart_index[i] = self.df.loc[end_idx+self.h, 'ghi_clear_sky']
            else:
                self.smart_index[i] = self.df.loc[end_idx+self.h, 'ghi_clear_sky']*self.df.loc[end_idx, 'ghi']/self.df.loc[end_idx, 'ghi_clear_sky']
        # correct smart index not too small or large
        self.smart_index = np.clip(self.smart_index, 0, self.df.loc[(self.P-1+self.h):(self.df.shape[0]-1),'GHI_daily_max_clearsky'].values)
        if self.data_flag == 'GHI_percent_wrt_max':
            self.smart_index = 100*self.smart_index/self.df.loc[(self.P-1+self.h):(self.df.shape[0]-1),'GHI_daily_max_clearsky'].values
            
    def _remove_nan(self):
        # remove nan from labels   
        nan_bool_data = np.any(np.isnan(self.rawdat), axis=(1,2))

        nan_bool_label = np.isnan(self.labels)
        nan_bool = nan_bool_data | nan_bool_label
        
        # remove index of self.times if it is not in self.image_times
        # (applied unconditionally so the ts and image models share the same test range)
        if self.align_image:
            # drop samples with no matching image timestamp
            not_image_bool = np.isin(self.times, self.image_times, invert=True)
            nan_bool = nan_bool | not_image_bool

        return np.where(~nan_bool)[0]
    
    def _zeroClearSky(self):
        zero_bool = self.labels == 0
        return np.where(zero_bool)[0]
    
    def _split_index(self, indices):
        self.train = int(0.7 * self.len)
        self.valid = int(0.9 * self.len)
        train_set = self.index[range(0, self.train)]
        valid_set = self.index[range(self.train, self.valid)]
        test_set = self.index[range(self.valid, self.len)]  
        all_test_set = self.index
        if indices != 'auto':
            # indices should be a list of 2: peirod needs to be excluded from the training set
            self.times_screened = self.times[self.index]
            bool_exclude = (self.times_screened >= pd.to_datetime(indices[0])) & (self.times_screened <= pd.to_datetime(indices[1]))
            index_exclude = self.index[bool_exclude]
     
            train_set = np.setdiff1d(train_set, index_exclude)
            test_set = np.union1d(test_set, index_exclude)
            all_test_set = np.setdiff1d(self.index, index_exclude)

            
        if not self.creat_real_test:
                # remove samples whose clear-sky GHI is 0
            train_set = np.setdiff1d(train_set, self.index_zeroClearSky)
            valid_set = np.setdiff1d(valid_set, self.index_zeroClearSky)
            test_set = np.setdiff1d(test_set, self.index_zeroClearSky)    
            all_test_set = np.setdiff1d(self.index, self.index_zeroClearSky)
        split_dict = {'train': train_set, 'val': valid_set, 'test': test_set, 'all_test':all_test_set}

        self.times = self.times[split_dict[self.flag]]
        self.rawdat = self.rawdat[split_dict[self.flag]]
        self.labels = self.labels[split_dict[self.flag]]
        self.smart_index = self.smart_index[split_dict[self.flag]]
    
    def _split_index_special(self, indices):
        # build the test set explicitly from the given indices.
        # the remaining data is split 80/20 into train/val.
        assert indices != 'auto'
        
        self.times_screened = self.times[self.index]
        bool_exclude = (self.times_screened >= pd.to_datetime(indices[0])) & (self.times_screened <= pd.to_datetime(indices[1]))
        test_set = self.index[bool_exclude]
    
        train_val_set = np.setdiff1d(self.index, test_set)
        self.train = int(0.8 * len(train_val_set))
        train_set = train_val_set[range(0, self.train)]
        valid_set = np.setdiff1d(train_val_set, train_set)

            
        if not self.creat_real_test:
                # remove samples whose clear-sky GHI is 0
            train_set = np.setdiff1d(train_set, self.index_zeroClearSky)
            valid_set = np.setdiff1d(valid_set, self.index_zeroClearSky)
            test_set = np.setdiff1d(test_set, self.index_zeroClearSky)    
        split_dict = {'train': train_set, 'val': valid_set, 'test': test_set}

        self.times = self.times[split_dict[self.flag]]
        self.rawdat = self.rawdat[split_dict[self.flag]]
        self.labels = self.labels[split_dict[self.flag]]
        self.smart_index = self.smart_index[split_dict[self.flag]]   
        
    def preprocess_image(self, T,device):
        image_idx = [np.where(self.image_times == t)[0][0] for t in T]
        X_img = torch.from_numpy(self.pixel_values[image_idx].astype(np.float32) / 255.0).to(device)
        if X_img.ndim == 3:
            X_img = X_img.unsqueeze(1)
        return X_img

    def get_batches(self, batch_size, device, shuffle=False,bootstrap_idx=[]):

        length = len(self.times)
        index = torch.arange(length)
        if len(bootstrap_idx) > 0:
            index = index[bootstrap_idx]
            
        if shuffle:
            index = index[torch.randperm(length)]
            
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            excerpt = np.atleast_1d(excerpt)# ensure a single element is still treated as an array
            
            X_ts = torch.from_numpy(self.rawdat[excerpt,:]).float().to(device)
            Y = torch.from_numpy(self.labels[excerpt]).float().to(device)
            T = self.times[excerpt]

            Reference = None
            if self.smart_token:
                Reference = torch.from_numpy(self.smart_index[excerpt]).float().to(device)
                
            if self.image_token and self.ts_token:
                # get corresponding image data
                X_img = self.preprocess_image(T,device)
                data_tuple = (X_img, X_ts, Y) + ((Reference, T) if self.smart_token else (T,))
            elif not self.image_token and self.ts_token:
                data_tuple = (X_ts, Y) + ((Reference, T) if self.smart_token else (T,))
            elif self.image_token and not self.ts_token:
                X_img = self.preprocess_image(T,device)  
                data_tuple = (X_img, Y) + ((Reference, T) if self.smart_token else (T,))
            yield data_tuple
            start_idx += batch_size


class DataGenerator_ViLT_2img(object):
    def __init__(self, image1, image_time1,image2, image_time2,ts_data,horizon, window,flag,meteo=False,data_flag='GHI_percent_wrt_max',
                 indices='auto',image_token = False, ts_token= False,smart_token=False,image_stack = False,creat_real_test=False,special_test=False):
        # what data to use
        self.image_token = image_token
        self.ts_token = ts_token
        self.smart_token = smart_token
        self.image_stack = image_stack
        if self.image_stack: # for 2img CNNLSTM model only
            assert self.image_token == True and self.ts_token == False
        # img data
        self.pixel_values = np.load(image1, mmap_mode='c', allow_pickle=True)
        if self.pixel_values.shape[-1] == 3 or self.pixel_values.shape[-1]==1:
            self.pixel_values = np.transpose(self.pixel_values,(0,3,1,2))
        self.image_times1 = np.load(image_time1,allow_pickle=True).astype('datetime64[s]')
        self.pixel_values2 = np.load(image2, mmap_mode='c', allow_pickle=True)
        if self.pixel_values2.shape[-1] == 3 or self.pixel_values2.shape[-1]==1:
            self.pixel_values2 = np.transpose(self.pixel_values2,(0,3,1,2))
        self.image_times2 = np.load(image_time2,allow_pickle=True).astype('datetime64[s]')
        
        self.image_times = np.intersect1d(self.image_times1,self.image_times2)
        # ts data
        self.P = window
        self.h = horizon
        self.flag = flag
        assert self.flag in ['train', 'val', 'test','all_test']
        self.creat_real_test = creat_real_test
        self.meteo_flag = meteo
        if meteo == True:
            self.meteo = ['Relative_air_humidity', 'Air_temperature','Wind_speed', 'Wind_direction', 'Gust_peak']
            #self.meteo.append(data_flag)
        else:
            self.meteo = [data_flag]
        self.data_flag = data_flag
        self._get_sequence(ts_data)
 
        #remove nan from ts data, from labels, and get zero clear sky index
        self.index = self._remove_nan()
        self.index_zeroClearSky = self._zeroClearSky()
        self.len = len(self.index)  

        # indices can be None or a dict of tuples (start, end) at length of 2
        #assert indices == 'auto' or (isinstance(indices, dict) and len(indices) == 3)  
        
        self._split_index(indices)

        if self.rawdat.ndim == 2:
            self.rawdat = np.expand_dims(self.rawdat, axis=2)
        self.rawdat = np.transpose(self.rawdat, (0, 2, 1))
        
    def _get_sequence(self,ts_data):
        self.df = pd.read_csv(ts_data)
        self.df['time'] = pd.to_datetime(self.df['time']).values.astype('datetime64[s]')
        
        n = self.df.shape[0]-self.h-self.P+1
        self.times = np.zeros((n),dtype='datetime64[s]')
        self.rawdat = np.zeros((n,self.P,len(self.meteo)))
        self.labels = np.zeros((n))
        self.smart_index = np.zeros((n))
        for i in range(n):
            end_idx = i+self.P-1
            self.times[i] = self.df.loc[end_idx, 'time']
            self.rawdat[i] = self.df.loc[end_idx-self.P+1:end_idx, self.meteo].values
            
            self.labels[i] = self.df.loc[end_idx+self.h, self.data_flag]
            # add smart index
            if np.isnan(self.df.loc[end_idx, 'ghi_clear_sky']):
                if np.isnan(self.df.loc[end_idx+self.h, 'ghi_clear_sky']):
                    self.smart_index[i] = np.nan
                else:
                    self.smart_index[i] = self.df.loc[end_idx+self.h, 'ghi_clear_sky']
            elif self.df.loc[end_idx, 'ghi_clear_sky'] == 0:
                self.smart_index[i] = self.df.loc[end_idx+self.h, 'ghi_clear_sky']
            else:
                self.smart_index[i] = self.df.loc[end_idx+self.h, 'ghi_clear_sky']*self.df.loc[end_idx, 'ghi']/self.df.loc[end_idx, 'ghi_clear_sky']
        # correct smart index not too small or large
        self.smart_index = np.clip(self.smart_index, 1, self.df.loc[(self.P-1+self.h):(self.df.shape[0]-1),'GHI_daily_max_clearsky'].values)
        if self.data_flag == 'GHI_percent_wrt_max':
            self.smart_index = 100*self.smart_index/self.df.loc[(self.P-1+self.h):(self.df.shape[0]-1),'GHI_daily_max_clearsky'].values
            
    def _remove_nan(self):
        # remove nan from labels   
        nan_bool_data = np.any(np.isnan(self.rawdat), axis=(1,2))

        nan_bool_label = np.isnan(self.labels)
        nan_bool = nan_bool_data | nan_bool_label
        
        # remove index of self.times if it is not in self.image_times
        # (applied unconditionally so the ts and image models share the same test range)
        not_image_bool = np.isin(self.times, self.image_times, invert=True)
        nan_bool = nan_bool | not_image_bool

        return np.where(~nan_bool)[0]
    
    def _zeroClearSky(self):
        zero_bool = self.labels == 0
        return np.where(zero_bool)[0]
    
    def _split_index(self, indices):
        self.train = int(0.7 * self.len)
        self.valid = int(0.9 * self.len)
        train_set = self.index[range(0, self.train)]
        valid_set = self.index[range(self.train, self.valid)]
        test_set = self.index[range(self.valid, self.len)]  
        if indices != 'auto':
            # indices should be a list of 2: peirod needs to be excluded from the training set
            self.times_screened = self.times[self.index]
            bool_exclude = (self.times_screened >= pd.to_datetime(indices[0])) & (self.times_screened <= pd.to_datetime(indices[1]))
            index_exclude = self.index[bool_exclude]
     
            train_set = np.setdiff1d(train_set, index_exclude)
            test_set = np.union1d(test_set, index_exclude)

            
        if not self.creat_real_test:
                # remove samples whose clear-sky GHI is 0
            train_set = np.setdiff1d(train_set, self.index_zeroClearSky)
            valid_set = np.setdiff1d(valid_set, self.index_zeroClearSky)
            test_set = np.setdiff1d(test_set, self.index_zeroClearSky)    

        split_dict = {'train': train_set, 'val': valid_set, 'test': test_set}

        self.times = self.times[split_dict[self.flag]]
        self.rawdat = self.rawdat[split_dict[self.flag]]
        self.labels = self.labels[split_dict[self.flag]]
        self.smart_index = self.smart_index[split_dict[self.flag]]
        
    def get_batches(self, batch_size, device, shuffle=False,bootstrap_idx=[]):

        length = len(self.times)
        index = torch.arange(length)
        if len(bootstrap_idx) > 0:
            index = index[bootstrap_idx]
            
        if shuffle:
            index = index[torch.randperm(length)]
            
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            excerpt = np.atleast_1d(excerpt)# ensure a single element is still treated as an array
            
            X_ts = torch.from_numpy(self.rawdat[excerpt,:]).float().to(device)
            Y = torch.from_numpy(self.labels[excerpt]).float().to(device)
            T = self.times[excerpt]

            Reference = None
            if self.smart_token:
                Reference = torch.from_numpy(self.smart_index[excerpt]).float().to(device)

            if self.image_token:
                # get corresponding image data
                image_idx1 = [np.where(self.image_times1 == t)[0][0] for t in T]
                image_idx2 = [np.where(self.image_times2 == t)[0][0] for t in T]
                X_img1 = torch.from_numpy(self.pixel_values[image_idx1].astype(np.float32)/255.0).to(device)
                X_img2 = torch.from_numpy(self.pixel_values2[image_idx2].astype(np.float32)/255.0).to(device)
                if X_img1.ndim == 3:
                    X_img1 = X_img1.unsqueeze(1)
                if X_img2.ndim == 3:
                    X_img2 = X_img2.unsqueeze(1)
                if self.image_stack:
                    # for 2img CNNLSTM model only
                    X_img = torch.stack((X_img1,X_img2),dim=1)
                    assert X_img.ndim == 5  
                    data_tuple = (X_img, Y) + ((Reference, T) if self.smart_token else (T,))
                else:
                    data_tuple = (X_img1,X_img2, X_ts, Y) + ((Reference, T) if self.smart_token else (T,))
            else:
                data_tuple = (X_ts, Y) + ((Reference, T) if self.smart_token else (T,))
            yield data_tuple
            start_idx += batch_size

class DataGenerator_imgs(object):
    def __init__(self, image, image_time,ts_data,horizon, flag,window=144,img_num=3,meteo=False,data_flag='GHI_percent_wrt_max',
                 indices='auto',image_token = False, smart_token=False,creat_real_test=False):
        # what data to use
        self.image_token = image_token
        self.smart_token = smart_token
        
        # img data
        self.pixel_values = np.load(image, mmap_mode='c', allow_pickle=True)
        if self.pixel_values.shape[-1] == 3 or self.pixel_values.shape[-1]==1:
            self.pixel_values = np.transpose(self.pixel_values,(0,3,1,2))
        self.image_times = np.load(image_time,allow_pickle=True).astype('datetime64[s]')

        self.img_num = img_num
        
        # ts data
        self.P = window
        self.h = horizon
        self.flag = flag
        assert self.flag in ['train', 'val', 'test']
        self.creat_real_test = creat_real_test
        self.meteo_flag = meteo
        if meteo == True:
            self.meteo = ['Relative_air_humidity', 'Air_temperature','Wind_speed', 'Wind_direction', 'Gust_peak']
            #self.meteo.append(data_flag)
        else:
            self.meteo = [data_flag]
        self.data_flag = data_flag
        self._find_continuous_segments()
        self._get_sequence(ts_data)
 
        #remove nan from ts data, from labels, and get zero clear sky index
        self.index = self._remove_nan()
        self.index_zeroClearSky = self._zeroClearSky()
        self.len = len(self.index)  

        # indices can be None or a dict of tuples (start, end) at length of 2
        #assert indices == 'auto' or (isinstance(indices, dict) and len(indices) == 3)  
        
        self._split_index(indices)

        if self.rawdat.ndim == 2:
            self.rawdat = np.expand_dims(self.rawdat, axis=2)
        self.rawdat = np.transpose(self.rawdat, (0, 2, 1))
    
    def _find_continuous_segments(self):
        continuous_indices = []
        for i in range(self.img_num-1, len(self.image_times)):
            if np.all(np.diff(self.image_times[i-self.img_num+1:i+1]) == np.timedelta64(10, 'm')):
                continuous_indices.append(i)
        # Return the continuous segments and their corresponding pixel values
        self.image_times = self.image_times[continuous_indices]
        self.pixel_values = self.pixel_values[continuous_indices]
    
    def _get_sequence(self,ts_data):
        self.df = pd.read_csv(ts_data)
        self.df['time'] = pd.to_datetime(self.df['time']).values.astype('datetime64[s]')
        
        n = self.df.shape[0]-self.h-self.P+1
        self.times = np.zeros((n),dtype='datetime64[s]')
        self.rawdat = np.zeros((n,self.P,len(self.meteo)))
        self.labels = np.zeros((n))
        self.smart_index = np.zeros((n))
        for i in range(n):
            end_idx = i+self.P-1
            self.times[i] = self.df.loc[end_idx, 'time']
            self.rawdat[i] = self.df.loc[end_idx-self.P+1:end_idx, self.meteo].values
            
            self.labels[i] = self.df.loc[end_idx+self.h, self.data_flag]
            # add smart index
            if np.isnan(self.df.loc[end_idx, 'ghi_clear_sky']):
                if np.isnan(self.df.loc[end_idx+self.h, 'ghi_clear_sky']):
                    self.smart_index[i] = np.nan
                else:
                    self.smart_index[i] = self.df.loc[end_idx+self.h, 'ghi_clear_sky']
            elif self.df.loc[end_idx, 'ghi_clear_sky'] == 0:
                self.smart_index[i] = self.df.loc[end_idx+self.h, 'ghi_clear_sky']
            else:
                self.smart_index[i] = self.df.loc[end_idx+self.h, 'ghi_clear_sky']*self.df.loc[end_idx, 'ghi']/self.df.loc[end_idx, 'ghi_clear_sky']
        # correct smart index not too small or large
        self.smart_index = np.clip(self.smart_index, 1, self.df.loc[(self.P-1+self.h):(self.df.shape[0]-1),'GHI_daily_max_clearsky'].values)
        if self.data_flag == 'GHI_percent_wrt_max':
            self.smart_index = 100*self.smart_index/self.df.loc[(self.P-1+self.h):(self.df.shape[0]-1),'GHI_daily_max_clearsky'].values
            
    def _remove_nan(self):
        # remove nan from labels   
        nan_bool_data = np.any(np.isnan(self.rawdat), axis=(1,2))

        nan_bool_label = np.isnan(self.labels)
        nan_bool = nan_bool_data | nan_bool_label
        
        # remove index of self.times if it is not in self.image_times
        # (applied unconditionally so the ts and image models share the same test range)
        not_image_bool = np.isin(self.times, self.image_times, invert=True)
        nan_bool = nan_bool | not_image_bool

        return np.where(~nan_bool)[0]
    
    def _zeroClearSky(self):
        zero_bool = self.labels == 0
        return np.where(zero_bool)[0]
    
    def _split_index(self, indices):
        self.train = int(0.7 * self.len)
        self.valid = int(0.9 * self.len)
        train_set = self.index[range(0, self.train)]
        valid_set = self.index[range(self.train, self.valid)]
        test_set = self.index[range(self.valid, self.len)]  
        if indices != 'auto':
            # indices should be a list of 2: peirod needs to be excluded from the training set
            self.times_screened = self.times[self.index]
            bool_exclude = (self.times_screened >= pd.to_datetime(indices[0])) & (self.times_screened <= pd.to_datetime(indices[1]))
            index_exclude = self.index[bool_exclude]
     
            train_set = np.setdiff1d(train_set, index_exclude)
            test_set = np.union1d(test_set, index_exclude)

            
        if not self.creat_real_test:
                # remove samples whose clear-sky GHI is 0
            train_set = np.setdiff1d(train_set, self.index_zeroClearSky)
            valid_set = np.setdiff1d(valid_set, self.index_zeroClearSky)
            test_set = np.setdiff1d(test_set, self.index_zeroClearSky)    

        split_dict = {'train': train_set, 'val': valid_set, 'test': test_set}

        self.times = self.times[split_dict[self.flag]]
        self.rawdat = self.rawdat[split_dict[self.flag]]
        self.labels = self.labels[split_dict[self.flag]]
        self.smart_index = self.smart_index[split_dict[self.flag]]
        
    def get_batches(self, batch_size, device, shuffle=False,bootstrap_idx=[]):

        length = len(self.times)
        index = torch.arange(length)
        if len(bootstrap_idx) > 0:
            index = index[bootstrap_idx]
            
        if shuffle:
            index = index[torch.randperm(length)]
            
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            excerpt = np.atleast_1d(excerpt)# ensure a single element is still treated as an array
            
            #X_ts = torch.from_numpy(self.rawdat[excerpt,:]).float().to(device)
            Y = torch.from_numpy(self.labels[excerpt]).float().to(device)
            T = self.times[excerpt]

            Reference = None
            if self.smart_token:
                Reference = torch.from_numpy(self.smart_index[excerpt]).float().to(device)

            if self.image_token:
                # get corresponding image data
                image_idxs = np.array([np.where(self.image_times == t)[0][0] for t in T])
                window_idxs = image_idxs[:, None] - np.arange(self.img_num)[::-1]
                X_imgs = torch.from_numpy(self.pixel_values[window_idxs].astype(np.float32)/255.0).to(device)
                data_tuple = (X_imgs, Y) + ((Reference, T) if self.smart_token else (T,))
                #data_tuple = (X_imgs, X_ts, Y) + ((Reference, T) if self.smart_token else (T,))
            else:
                data_tuple = (Y,) + ((Reference, T) if self.smart_token else (T,))
                #data_tuple = (X_ts, Y) + ((Reference, T) if self.smart_token else (T,))
            yield data_tuple
            start_idx += batch_size


class Data_provider_Unet(Dataset):
    def __init__(self, image, image_time, flag,train_only=False,indices='auto',
                 noise_level=0.001,apply_augmentation=False):
        
        # load image data
        self.pixel_values = np.load(image, mmap_mode='c', allow_pickle=True)
        self.image_time = np.load(image_time,allow_pickle=True).astype('datetime64[s]')
        if self.pixel_values.shape[-1] == 3 or self.pixel_values.shape[-1]==1:
            self.pixel_values = np.transpose(self.pixel_values,(0,3,1,2))

        # Image enhancements flags
        self.noise_level = noise_level
        self.apply_augmentation = apply_augmentation

        # Splitting train/val/test
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.train_only = train_only
        
        if indices == 'auto':
            num_train = int(len(self.image_time) * (0.7 if not self.train_only else 1))
            num_vali = int(len(self.image_time) * 0.19)
            border1s = [0, num_train, num_train + num_vali]
            border2s = [num_train, num_train + num_vali, len(self.image_time)]
        else:
            border1s = [indices['train'][0], indices['val'][0], indices['test'][0]]
            border2s = [indices['train'][1], indices['val'][1], indices['test'][1]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        self.image_time = self.image_time[border1:border2]
        self.pixel_values = self.pixel_values[border1:border2]
        # self.pixel_values should be float instead of uint8 (input stays uint8; get_batches converts to float)
        assert self.pixel_values.dtype == np.uint8
        
        
    def add_noise(self, image):
        assert image.ndim == 4
        if image.dtype == np.uint8: 
            image = image/ 255.0
        # Adding random Gaussian noise within 0 to 1 range
        noise = np.random.normal(0, self.noise_level, image.shape)
        noisy_images = np.clip(image + noise, 0, 1)  # keep the noisy image within [0, 1]
        if image.dtype == np.uint8: 
            noisy_images = (noisy_images * 255).astype(np.uint8)
        return noisy_images
    
    def enhance_contrast_method(self, images):
        assert images.ndim == 4
        if images.dtype == np.uint8: 
            images = images/ 255.0
        # Enhancing contrast for images in 0 to 1 float range
        enhanced_images = np.empty_like(images)
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):  
                # RGB images in [0, 1]: histogram-equalise via skimage.exposure
                enhanced_images[i, j] = exposure.equalize_adapthist(images[i, j])
        if images.dtype == np.uint8:
            enhanced_images = (enhanced_images * 255).astype(np.uint8)
        return enhanced_images
    
    def augment_images(self, images):
        # Randomly apply different augmentations
        if random.choice([True, False]):
            images = self.add_noise(images)
        if random.choice([True, False]):
            images = self.enhance_contrast_method(images)
        return images
    
    def get_batches(self, batch_size, device, shuffle=False):
        length = len(self.image_time)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.arange(length)
        
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            
            if end_idx-start_idx == 1:
                X = self.pixel_values[excerpt].reshape(1, *self.pixel_values[excerpt].shape)
            else:
                X = self.pixel_values[excerpt]

            Y = X.copy()
            if self.apply_augmentation:
                X = self.augment_images(X)

                
            X = torch.from_numpy(X/255.).float()
            Y = torch.from_numpy(Y/255.).float()
            T = self.image_time[excerpt]
            
            # X = np.clip(X, 0, 1)
            # Y = np.clip(Y, 0, 1)
            yield X.to(device), Y.to(device), T

            start_idx += batch_size