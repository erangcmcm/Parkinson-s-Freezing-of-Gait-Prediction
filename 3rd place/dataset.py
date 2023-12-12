import os
import numpy as np
import pandas as pd
import random
import inspect
import datetime
import zstandard as zstd
import pickle

import cv2
import torch

from torch.utils.data import Dataset

import math
import scipy.signal as signal
from librosa.feature import melspectrogram

import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(datetime.datetime.now().microsecond)
np.random.seed(datetime.datetime.now().microsecond)

def getParams(cls, params):
    p = {k: params[k]
        for k in set(inspect.getfullargspec(cls).args) 
                    & set(params.keys())}
    print(p)
    return p


class WalkDataset(Dataset):
    def __init__(self, lcount, metadata, load, loadDaily, daily_split, 
                 seq = -1, patch = -1, test = True, id_frac = {},
                crop_rate = 0.3, excise_rate = 0.1, ablation_rate = 0.1,
                resize_rate = 0.9, stretch_rate = 0.5, 
                n_ablate = 1, n_excise = 1, 
                # gn = 0.0, 
                # sample_freq = 4,
                min_frac = 0.1, max_crop = 0.9, max_ablate = 0.3, max_excise = 0.3,
                lr_flip_rate = 0.5,
                    amp_gn = 0.3, amp_ch_gn = 0.1,
                    amp_gna = 0.1, amp_ch_gna = 0.03,

                    melspec = False,
                    n_mels = 16, #128
                    fft_mult = 2., # 1, 1.5, 2,
                    sr_mult = 4, #1, 2, 4, 8
                    mel_eps = 3e-6,
                    mel_pwr = 0.3,
                    mel_power = 2,
                    mel_patch = True,
                    mel_win_mult = 1.5, 

                    neg_mult = 0.0,

                    relabel = False,

         ):
        super().__init__()
        self.load = load
        self.loadDaily = loadDaily
        self.daily_split = daily_split
        self.id_frac = id_frac

        self.seq = seq
        self.patch = patch
        self.lcount = lcount
        self.metadata = metadata
        self.seqs = []
        self.seqlen = self.seq * self.patch
        seqlen = self.seqlen        
        
        self.test = test
        self.crop_rate = crop_rate
        self.resize_rate = resize_rate
        self.stretch_rate = stretch_rate
        # self.gn = gn
        self.amp_gn = amp_gn
        self.amp_ch_gn = amp_ch_gn
        self.amp_gna = amp_gna
        self.amp_ch_gna = amp_ch_gna
        self.min_frac = min_frac
        self.max_crop = max_crop
        self.max_ablate = max_ablate
        self.max_excise = max_excise
        self.n_ablate = n_ablate
        self.n_excise = n_excise    

        self.lr_flip_rate = lr_flip_rate

        self.ablation_rate = ablation_rate
        self.excise_rate = excise_rate

        self.melspec = melspec
        self.n_mels = n_mels 
        self.fft_mult = fft_mult
        self.sr_mult = sr_mult
        self.mel_eps = mel_eps
        self.mel_pwr = mel_pwr
        self.mel_power = mel_power
        self.mel_patch = mel_patch
        self.mel_win_mult = mel_win_mult

        self.neg_mult = neg_mult

        self.rng = np.random.default_rng()

        self.relabeling = False

        if daily_split > 0 and relabel and not test:
            s = daily_split
            self.relabeling = True
        else:
            self.relabeling = False
            s = seqlen // (1 if daily_split > 0
                        else 4 if not test else 2)
            
        for k, l in lcount.items():
            if 'unlabeled' not in k:
                v = load(k)
                assert l == v.shape[0]
                assert v.shape[1] in [12]#[6, 8]
            else:
                v = loadDaily(k, 0, 1000)
                assert v.shape[1] in [3, 7]
                pass;
            for i in ( range(-math.ceil(seqlen / s) if not self.relabeling else 0
                                , math.ceil(l / s)) ):
                self.seqs.append((k, i * s, (i + 1) * s, 
                                    max(0, i * s),
                                    min(l, (i + 1) * s)))
        self.cache = {}        

        # defaults to test is zero, otherwise mean; for unlabeled, for now;
        # to avoid overfiting on years/question score that are very different;
        m = np.zeros((self.metadata.shape[1], ), dtype = np.float32)
        if 'unlabeled' in k:
            m[11] = metadata.iloc[:, 11].min()
        self.default_metadata = m
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq, patch, seqlen = self.seq, self.patch, self.seqlen
        while True:
            f, start, end, seqmin, seqmax = self.seqs[idx]
            # print(f, start, end, seqmin, seqmax)
            
            # donwsample certain ids, specified
            sample_rate = self.id_frac.get(f.split('/')[-1].split('.')[0], 1)
            if random.random() > sample_rate: 
                # print(f, 'skipped'); 
                idx = random.randrange(len(self.seqs)); continue
            
            sidx = ( random.randrange(start, end) if not self.test else 
                        (start + end) // 2 + random.randrange(0, self.patch) - self.patch//2 )

            # i = random.randrange(start, end) if not self.test else seqmin#0# (start + end) // 2
            if sidx >= seqmax or sidx <= -self.seqlen:
                idx = random.randrange(len(self.seqs)); continue
            else:
                # print(f)
                if 'unlabeled' not in f:
                    # with cache;
                    c = self.cache.get(f)
                    if c is not None: v = c.copy(); #print('hit')
                    else:
                        v = self.load(f); #print('miss'); print(f)
                        if self.test:# use cache only at test time
                            if len(self.cache) > 10: del self.cache; self.cache = {}
                            self.cache[f] = v

                    assert sidx < v.shape[0]
                    flen = len(v)
                else:
                    l = (self.seqlen * (4 if not self.test else 1)
                            ) if not self.relabeling else self.daily_split
                    sidx = ( seqmin if self.test or self.relabeling else max(sidx, 0) )
                    k = (f, sidx//self.daily_split, (sidx + l - 1)//self.daily_split)
                    c = self.cache.get(k)
                    if c is not None: v = c; 
                    else:
                        v = self.loadDaily(f, sidx, sidx + l); 
                        if self.test:
                            if len(self.cache) > 20: self.cache = {}
                            self.cache[k] = v
                    offset = (sidx // self.daily_split ) * self.daily_split
                    flen = sidx + l
                    # print(len(v))
                # print(v.shape)
                # resize
                vlen = min(self.seqlen, v.shape[0] - sidx - 1, sidx + self.seqlen
                                ) if 'unlabeled' not in f else (
                    v.shape[0] + offset - sidx - 1)
                
                if vlen < self.seqlen * self.min_frac or vlen < 20: 
                    # print('below cutoff')
                    idx = random.randrange(len(self.seqs)); continue
                
                if self.relabeling:
                    srate = self.daily_split / self.seqlen * np.clip(v[:, 6] , 0, None).mean()
                    if random.random() > srate: 
                        # print(srate, 'rerdraw')
                        idx = random.randrange(len(self.seqs)); continue
                    else:
                        pass
                        # print(srate)
                
                resizing = random.random() < self.resize_rate and not self.test 
                ratio = np.exp(np.random.normal(0, self.stretch_rate)) if resizing else 1
                prelen = int(self.seqlen / ratio) * 4
                # print(start, end, seqmin, seqmax, sidx, v.shape[0])
                
                # test set starts with original window;
                # training starts wtih the random point selected within the window;
                # and crop a very large portion afterwards, sufficient for rescaling, crop, etc.;
                # note: SIDX is key variable used at test time to store preds;
                # if 'unlabeled' not in f: # 
                sidx = seqmin if self.test else max(sidx, 0)
                if self.relabeling: # redraw for density                    
                    wt = v[::10, 6]  ** 1.5
                    idx = np.random.choice(len(wt), p = wt / wt.sum()) * 10
                    idx -= self.seqlen // 2 + random.randrange(0, 10)
                    sidx = max(0, idx)                    
                    sidx += offset
                    
                iidx = sidx - (offset if 'unlabeled' in f else 0)
                xmin, xmax = iidx, min(iidx + int(prelen * (1 + self.max_crop)), v.shape[0])
                v = v[xmin: xmax]
                # print(v.shape)
                # else:                    
                #     xmin, xmax = sidx, sidx + len(v)
                #     flen = xmax
                    # print(xmin, xmax, flen)

                # calculate all positional features, based on origina indexing, i.e. 
                #  flen is length, xmin: xmax are the positional indices of array
                # and frac, lidx, ridx are the three columns
                if 'unlabeled' in f:
                    vlabeled = v.shape[1] > 6
                    if ( self.relabeling and   # qscore 0 --> reduce targets
                            self.metadata.loc[f.split('/')[-1].split('.')[0], 
                                              'NFOGQ'] == 0 ):
                        v[:, 3:6] *= self.neg_mult
                    v = np.concatenate((v, np.zeros((v.shape[0], 12 - v.shape[1]), 
                                    dtype = v.dtype)), 1)
                    if vlabeled:
                        v[:, 7] = v[:, 6] 

                vfull = np.concatenate([v, np.zeros((len(v), 3), dtype = v.dtype)], 1)
                vfull[:, -3] = np.linspace(xmin/flen, xmax/flen, len(v), dtype = v.dtype)                
                vfull[:, -2] = np.arange(xmin, xmax)
                vfull[:, -1] = flen - np.arange(xmin, xmax) - 1 
                v = vfull
                # print('  ', vfull[0, -3], vfull[-1, -3], vfull.shape)
                # print('  ', vfull.min(0)[-3:].astype(int), )
                # print('  ', vfull.max(0)[-3:].astype(int), )
                if resizing: v = cv2.resize(v, None, fx = 1, fy = ratio)
                
                # if not any valid/test-- draw another one; (after resizing)
                if  ('unlabeled' not in f   #   labeled data with no labels 
                       and random.random() > 0.2
                        and (v[:self.seqlen, 6:8].mean(1) == 1).sum(0) == 0  ):                     
                    idx = random.randrange(len(self.seqs)); continue
                
                elif ( 'unlabeled' in f and vlabeled and self.relabeling
                      and v[:self.seqlen, 6].mean() < 0.01 ):
                    idx = random.randrange(len(self.seqs)); continue                

                break # if didn't continue;
        

        # occasionally crop
        if not self.test:                
            crop = random.random() < self.crop_rate and not self.test and v.shape[0] > 3
            if len(v) > 10 and crop:
                if random.random() < 0.5:
                    v = v[:random.randrange(1, int(self.max_crop * min(len(v), self.seqlen)))]
                else:
                    v = v[-random.randrange(1, int(self.max_crop * min(len(v), self.seqlen))):]

        # trim
        v = v[:int(self.seqlen * (1 + self.max_excise * 1))]
        
        # g-scale;
        v[:, :3] *= (9.8 if '/tdcs' not in f else 1)/3

        # only to handle dead series and rounding error-- 1e-3, scaled back up;
        # v[:, :3] += self.rng.normal(0, 1e-3, v[:, :3].shape) * (9.8/3)

        # print(v.shape)
        # amplitude agn multiplier -- to raw sigmal
        if not self.test:
            gn = ( 
                    self.rng.normal(0, self.amp_gn, (1, ), )
                    + self.rng.normal(0, self.amp_ch_gn, (3, ) )
                    + self.rng.normal(0, self.amp_gna, (len(v), 1, ), ).cumsum(0) / len(v) ** 0.5
                    + self.rng.normal(0, self.amp_ch_gna, (len(v), 3, ) ).cumsum(0) / len(v) ** 0.5
            )
            v[:, :3] *= np.exp(gn)                            

        # TTA -- lr flip (!)
        if random.random() < self.lr_flip_rate:
            v[:, 1] *= -1


        # excise/ablation
        if not self.test and len(v) > 10 and not crop:
            # norm before ablate/excise;
            if len(v) > 0:
                v[:, :3] -= v[:, :3].mean(0)
            
            # len of abalation, and where is starts;
            l = random.randint(1, int(len(v) * self.max_excise ) )
            i = random.randint(0, len(v) - l)
            if random.random() < self.excise_rate:
                v = np.concatenate([v[:i],  # the 50/150 may not be tested, but w/e;
                                    np.zeros((random.randint(50, 150), v.shape[1]), dtype = v.dtype), 
                                    v[i + l:]])
            elif random.random() < self.ablation_rate:
                v[i: i + l, ] = 0

        # if not self.test:
        #     if self.gn:
        #         v[:, :3] += self.rng.normal(0, self.gn, v[:, :3].shape)


        # crop, normalize(mean) and pad/fill
        v = v[:self.seqlen]; 
        if len(v) > 0:
            v[:, :3] -= v[:, :3].mean(0)

        # print('  ', sidx, len(v))
        if v.shape[0] < self.seqlen:
            v = np.concatenate([v, np.zeros((self.seqlen - v.shape[0], v.shape[1]), dtype = np.float32)])

        # patchify and melspec;
        xp = v[:, :3].reshape(v.shape[0] // patch, patch, 3)
        if self.melspec:
            x = v[:, :3]
            m = melspectrogram(    y = np.pad(x, ((patch//2, patch - patch//2), (0, 0))).T, 
                    sr = 100, 
                    n_mels = self.n_mels, 
                    hop_length = patch,            
                    n_fft = 2 ** math.ceil(np.log2(patch * self.mel_win_mult * 1 )),
                    win_length = int(patch * self.mel_win_mult),
                    power = self.mel_power,
                    window = 'hann', center = True,
                    )[:, :, 1:-1] 
            m = (m ** self.mel_pwr ) if self.mel_pwr else np.log(m + self.mel_eps) / 3
            m = m.transpose(2, 1, 0)
            x = np.concatenate((m, xp), 1) if self.mel_patch else m
        else:
            x = xp 

        # targets, weights, masks, metadata, etc.
        y = v[:, 3:-3] if v.shape[1] > 3 else np.zeros((v.shape[0], 8), dtype = np.float32)
        assert y[:, :3].sum(1).max() <= 1 and y[:, :-1].max() <= 1

        s = np.float32(1.0) * ( (v[:, 6:8].mean(1) > 1 - 1e-5)
                                if not self.relabeling  
                                 else v[:, 6:8].mean(1)
                                  )


        # if task targets are non-int--repair;
        for i in range(10):
            frac_t = y[:, -1] % 1 > 0; 
            if frac_t.sum() == 0: break
            left, right = np.roll(y[:, -1], 1), np.roll(y[:, -1], -1)
            y[:, -1] = np.where(frac_t, np.where(left % 1 == 0, left, right), y[:, -1]) 
            # if i > 2: print('{}th fraction fill for {}'.format(i, f))

        frac = v[:, -3:] * (0 if 'unlabeled' in f else 1)
        _id = f.split('/')[-1].split('.')[0]
        if 'unlabeled' not in f:# self.metadata.index:
            m = self.metadata.loc[_id].values
        else:
            m = self.default_metadata
    
        return x, y, s, frac, m, f, sidx, flen