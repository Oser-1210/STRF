from scipy import stats, signal
from scipy.stats import zscore
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.linalg as la
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit


class STRF():
    
    def __init__(self, n_components=4, n_band=5, montage=20, srate=250, fs_stimulus=60, T_past=1, T_futu=0.14, l_min = 0.04, l_max = 0.2) -> None:
        
        self.n_components = n_components
        self.n_band = n_band
        self.montage = montage
        self.srate = srate
        self.fs_stimulus = fs_stimulus
        self.T_past = T_past
        self.T_futu = T_futu
        self.l_min = int(l_min*srate)
        self.l_max = int(l_max*srate)
        
        pass
            
    def preprocess(self, X):

        filteredX = []
        for epoch in X:

            filteredEpoch = self.filterbank(epoch, self.srate, 0) # using the first sub-band 
            filteredX.append(filteredEpoch)
        filteredX = np.stack(filteredX)

        return filteredX
    
    def Upresample(self, S):
    
        S_resample = np.zeros((S.shape[0], int(self.srate/self.fs_stimulus*S.shape[1]), S.shape[2], S.shape[3]))
        
        for i in range(S_resample.shape[0]):
            for j in range(S_resample.shape[1]):
                
                num = math.floor(j/self.srate*self.fs_stimulus)
                S_resample[i,j] = np.squeeze(S[i,num,:,:])
        return S_resample
    
    def filterbank(self, x, srate, freqInx):

        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]

        srate = srate/2
        Wp = [passband[freqInx]/srate, 90/srate]
        Ws = [stopband[freqInx]/srate, 100/srate]
        [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)
        [B, A] = signal.cheby1(N, 0.5, Wn, 'bandpass')

        filtered_signal = np.zeros(np.shape(x))
        if len(np.shape(x)) == 2:
            for channelINX in range(np.shape(x)[0]):
                filtered_signal[channelINX, :] = signal.filtfilt(
                    B, A, x[channelINX, :])
            filtered_signal = np.expand_dims(filtered_signal, axis=-1)
        else:
            for epochINX, epoch in enumerate(x):
                for channelINX in range(np.shape(epoch)[0]):
                    filtered_signal[epochINX, channelINX, :] = signal.filtfilt(
                        B, A, epoch[channelINX, :])

        return filtered_signal
    
    def tdca_filter(self, X, y):
        # X: 120*19*750
        self.classes = np.unique(y)
        self.filters = []
        self.epochNUM, self.channelNUM, _ = X.shape
        augumentX = []
        for classINX in self.classes:

            this_class_data = X[y == classINX]
            augumentEpoch = []
            for epoch in this_class_data:

                augumentEpoch.append(epoch)
            augumentX.append(np.stack(augumentEpoch))
        # TDCA weight
        # augmentX = 20*6*19*750
        self.computer_tdca_weight(augumentX)
        # TDCA filter
        tdca_X = []
        for comp in self.n_components:

            tdca_class_X = []
            for classX in augumentX:
                
                tdca_epoch_X = []
                for epochX in classX:
                    
                    filteredX = np.squeeze(np.dot(epochX.T, self.filters[comp]))
                    tdca_epoch_X.append(filteredX)
                tdca_class_X.append(np.stack(tdca_epoch_X))
            tdca_X.append(np.stack(tdca_class_X))
        tdca_X = np.stack(tdca_X)
        tdca_X = np.mean(tdca_X, axis=2)
       
        return tdca_X
    
    def computer_tdca_weight(self, X):
        # X = 20*6*19*750 classX = 20*19*750
        classX = np.mean(X, axis=1)

        classX = classX - np.mean(classX, axis=-1, keepdims=True)
        betwClass = classX - np.mean(classX, axis=0, keepdims=True)
        betwClass = np.concatenate(betwClass, axis=1)
        # within classes
        X = X - np.mean(X, axis=-1, keepdims=True)
        withinClass = X - np.mean(X, axis=1, keepdims=True)
        withinClass = np.concatenate(withinClass, axis=2)
        withinClass = np.concatenate(withinClass, axis=1)
        
        Hb = betwClass/math.sqrt(self.montage)
        Hq = withinClass/math.sqrt(self.epochNUM)
        Sb = np.dot(Hb, Hb.T)
        Sq = np.dot(Hq, Hq.T)
        
        C = np.linalg.inv(Sb).dot(Sq)
        # _, W = np.linalg.eig(C)
        Diag, W = la.eig(C)
        index = np.argsort(Diag)
        W = W[:,index[:self.n_components]]
        W = W * np.sign(W[np.argmax(abs(W),axis=0)])
        self.filters.append(W)
           
        return
    
    def fit(self, data, S):
        
        X = data['X'][:,:,int(self.T_futu*self.srate):]
        X = self.preprocess(X)
        y = data['y']
        channels = data['channel']

        S = self.Upresample(S)
        S = np.flip(S, axis=2)
        STRFs = {}
        RMSmap = {}
        Gauss = {}

        for chnINX, chnName in enumerate(channels):

            chnX = np.squeeze(X[:,chnINX,:])
            Kz = 0
            K = 0
            for epochINX, epoch in enumerate(chnX):

               stim = S[y[epochINX]-1]
               K_raw, K_norm = self.reverse_correlation(epoch, stim)
               Kz += K_norm
               K += K_raw
            Kz = Kz/chnX.shape[0]
            K = K/chnX.shape[0]
            
            RMS = self.RMSmap(Kz)
            try:
                gauss_param = self.fit_gaussian(RMS)
            except:
                print('Fit failed!')
                gauss_param = []

            STRFs[chnName] = Kz
            # STRFs[chnName] = K
            RMSmap[chnName] = RMS
            Gauss[chnName] = gauss_param
        self.STRFs = STRFs
        
        return STRFs, RMSmap, Gauss
    
    def fit_only(self, data, S, seed=None):
        
        X = data['X'][:,:,int(self.T_futu*self.srate):]
        X = self.preprocess(X)
        y = data['y']
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(y)
        channels = data['channel']

        S = self.Upresample(S)
        S = np.flip(S, axis=2)
        STRFs = {}

        for chnINX, chnName in enumerate(channels):

            chnX = np.squeeze(X[:,chnINX,:])
            Kz = 0
            K = 0
            for epochINX, epoch in enumerate(chnX):

               stim = S[y[epochINX]-1]
               K_raw, K_norm = self.reverse_correlation(epoch, stim)
               Kz += K_norm
               K += K_raw
            Kz = Kz/chnX.shape[0]
            K = K/chnX.shape[0]
            STRFs[chnName] = Kz
        self.STRFs = STRFs
        
        return STRFs
    
    def fit_tdca(self, data, S, seed=None):

        X = data['X'][:,:,int(self.T_futu*self.srate):]
        X = self.preprocess(X)
        y = data['y']
        X = self.tdca_filter(X,y)
        y = np.unique(y)
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(y)

        S = self.Upresample(S)
        S = np.flip(S, axis=2)
        STRFs = {}

        for cinx in range(self.n_components):

            compX = np.squeeze(X[cinx,:,:])
            Kz = 0
            K = 0
            for epochINX, epoch in enumerate(compX):

               stim = S[y[epochINX]-1]
               K_raw, K_norm = self.reverse_correlation(epoch, stim)
               Kz += K_norm
               K += K_raw
            Kz = Kz/compX.shape[0]
            K = K/compX.shape[0]
            text = 'comp'+ str(cinx+1)
            STRFs[text] = Kz

        self.STRFs = STRFs


    
    def gaussian_2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
        return offset + amplitude * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))
        )
    
    def gaussian_2d_wrapper(self, coords, x0, y0, sigma_x, sigma_y, amplitude, offset):
        x, y = coords
        return self.gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset).ravel()
    
    def fit_gaussian(self, data):

        row, column = data.shape
        x = np.linspace(0,row-1,row)
        y = np.linspace(0,column-1,column)
        x, y = np.meshgrid(x, y)

        initial_guess = ((row-1)/2, (column-1)/2, 2, 2, np.max(data), np.min(data))
        popt, pcov = curve_fit(self.gaussian_2d_wrapper, (x, y), data.ravel(), p0=initial_guess, bounds=([row*0.1,column*0.1,0,0,0,0],[row*0.9,column*0.9,row/2,column/2,10,10]))
        (x0, y0, sigma_x, sigma_y, amplitude, offset) = popt
        fwhm_x = 2 * np.sqrt(2 * np.log(2)) * abs(sigma_x)
        fwhm_y = 2 * np.sqrt(2 * np.log(2)) * abs(sigma_y)
        size = (fwhm_x+fwhm_y)/2
        loc_x = x0-(row-1)/2
        loc_y = y0-(column-1)/2

        return loc_x, loc_y, x0, y0, size
    
    def RMSmap(self, Kz):

        kernal = np.flip(Kz, axis=0)
        kernal = kernal[self.l_min:self.l_max]      
        kernal_square = np.square(kernal)
        RMS = kernal_square.mean(axis=0)
        RMS = np.sqrt(RMS)

        return RMS

  

    def reverse_correlation(self, data, noise_stimulus):

        time_axis = np.arange(data.shape[-1])/self.srate
        min_t     = np.floor((self.T_past*self.srate)).astype(int)/self.srate # minimum time so that there is enough time "in the past"
        max_t     = np.floor((noise_stimulus.shape[0]/self.srate - self.T_futu)*self.srate).astype(int)/self.srate # maximum time so that there is enough data "in the future"

        select_mask = np.logical_and(time_axis > min_t, time_axis < max_t)
        time_axis   = time_axis[select_mask] # restrict time axis
        data_values = data[select_mask]

        S_futu = int(self.T_futu*self.srate)
        S_past = int(self.T_past*self.srate)
        K = np.zeros((S_futu+S_past,)+ noise_stimulus.shape[1:])# initialize kernel

        for i in range(len(time_axis)):
            
            t = time_axis[i]
            frame_in_stimulus = int(t*self.srate)
            history = noise_stimulus[frame_in_stimulus-S_past:frame_in_stimulus+S_futu, :, :]
            K += history * data_values[i]

        K_raw  = K/float(len(time_axis))
        K_norm = (K-K.mean())/K.std()

        return K_raw, K_norm
    
    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    def extractSTRF(self, R):

        '''
        takes an array 3-dim array R (x,y,time) containing the reverse correlation output and do first analysis:
        returns
        maxima, maxima_i, maxima_j, maxima_t

        maxima: the extreme (min or max) value within each RF
        maxima_i: the i-coordinate of the maxima
        maxima_j: the j-coordinate of the maxima
        maxima_t: the time-coordinate of the maxima

        '''
        maxmum = np.absolute(R).argmax()
        i,j,z = np.unravel_index(maxmum, R.shape)
        max_value = R[i,j,z]

        maxima_t = i
        maxima_x = j
        maxima_y = z
        maxima = max_value

        return maxima, maxima_x, maxima_y, maxima_t
    
    def gaussian_2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
        return offset + amplitude * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))
        )

    # 2D高斯函数的包装器，用于curve_fit
    def gaussian_2d_wrapper(self, coords, x0, y0, sigma_x, sigma_y, amplitude, offset):
        x, y = coords
        return self.gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset).ravel()