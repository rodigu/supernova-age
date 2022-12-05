from read_in_data_code import plot_snana_fits
import matplotlib.pyplot as plt
import sncosmo
import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas

NUM_DAYS = 2 # modify this to check results with 1 day precision

def get_data():
    models = os.listdir('data/')[1::]
    heads_list = []
    phots_list = []
    for model in models:
        heads = sorted(glob.glob(os.path.join(f'data/{model}', '*_HEAD.FITS.gz')))
        phots = sorted(glob.glob(os.path.join(f'data/{model}', '*_PHOT.FITS.gz')))
        assert len(heads) != 0, 'no *_HEAD_FITS.gz are found'
        assert len(heads) == len(phots), 'there are different number of HEAD and PHOT files'
        # files = os.listdir(f'data/{model}')
        num_heads = len(heads)
        # range_for_files = range(0,num_heads)
        random_files = np.random.choice(num_heads, size = 100,replace = False)
        for r in random_files:
            heads_list.append(heads[r])
            phots_list.append(phots[r])
    # print(len(heads_list),heads_list)
    # print(len(phots_list),phots_list)
    appended_data = -1

    # skip_size = 100
    for head, phot in zip(heads_list[::], phots_list[::]): #lots of LCs per head, phot files, so do a few to start
            #for head, phot in zip(heads_list[::skip_size], phots_list[::skip_size]):
        i = head.find('_HEAD.FITS.gz')
        assert head[:i] == phot[:i], f'HEAD and PHOT files name mismatch: {head}, {phot}'
        filename = head[:i].split('/')[1:3]#.split('.')[0:2]
        # num_heads = 200
        print(f'Current File: {filename}')
        for LCnum, lc in enumerate(sncosmo.read_snana_fits(head, phot)):#[0:num_heads]): # remember: multiple SN in single HEAD/PHOT file
            lc_meta = {lc.meta['SNID']:lc.meta}
            
            print(f'LCnum: {LCnum}')
            #print(lc.columns)
            lc_df = lc.to_pandas()
            
            # lc_df_meta = pandas.DataFrame.from_dict(lc_meta,orient='columns')

            lc_df['SNID']=lc.meta['SNID']
            lc_df['1stDet']=lc.meta['MJD_DETECT_FIRST']
            lc_df['Trigger']=lc.meta['MJD_TRIGGER']
            lc_df['Model_num']=lc.meta['SIM_TYPE_INDEX']
            # lc_df['filename']=filename[1]
            
            # fig, ax = plt.subplots()
            lc_df['BAND']= lc_df['BAND'].str.decode("utf-8") # turn Bytes into str
            lc_df['SNR'] = lc_df['FLUXCAL']/lc_df['FLUXCALERR'] # signal to noise ratio
            lc_df['MAG'] = np.array(-2.5*np.log10(np.abs(lc_df['FLUXCAL'])))+27.5 # magnitudes
            lc_df['MAGERR'] = 1.086/lc_df['SNR'] # magnitude error
            
                    # observations              # nondetections           # Signal to noise cut       
            mask = ((lc_df['PHOTFLAG'] != 0) | (lc_df['SNR'] >= 4))  #| (lc_df['PHOTFLAG'] == 0)  | (lc_df['SNR'] >= 4) 
            lc_df = lc_df[mask].reset_index(drop=True)
            
            D_id_color = {
                    "X ": u"#b9ac70", # ZTF-g
                    "Y ": u"#bd1f01", # ZTF-r
                    "g ": u"#4daf4a", # YSE-g
                    "r ": u"#e41a1c", # YSE-r
                    "i ": u"#832db6", # YSE-i
                    "z ": u"#656364"} # YSE-z
            lc_df['PLOTCOLOR'] = lc_df.BAND.map(D_id_color)
            
            if type(appended_data) == int:
                appended_data = lc_df.copy()
            else:
                appended_data = pandas.concat([appended_data, lc_df])
    
    return masked_data(appended_data)

def masked_data(df):
    df['int_MJD'] = df['MJD'].astype(int)
    df['norm_MJD'] = (df['MJD'] / NUM_DAYS).astype(int)
    df['BAND_r'] = df.apply(lambda row: np.NaN if row['BAND'] != 'r ' else row['MAG'],axis=1)
    df['BAND_g'] = df.apply(lambda row: np.NaN if row['BAND'] != 'g ' else row['MAG'],axis=1)
    df['BAND_i'] = df.apply(lambda row: np.NaN if row['BAND'] != 'i ' else row['MAG'],axis=1)
    return df
    # # print('Unique SNID sample: ', df['SNID'].nunique())
    # lc_df_int_MJD = df.groupby(['SNID', 'norm_MJD']).mean(['BAND_r','BAND_i','BAND_g']).reset_index()
    # band_mask = (~lc_df_int_MJD['BAND_r'].isna()) & (~lc_df_int_MJD['BAND_g'].isna()) & (~lc_df_int_MJD['BAND_i'].isna())
    # return df[df['norm_MJD'].isin(lc_df_int_MJD[band_mask]['norm_MJD'])]

def average_bands(df):
    snid_df = df.groupby('SNID').mean(numeric_only=True)
    return snid_df.dropna()

def run_pipeline(days_range=2):
    global NUM_DAYS
    NUM_DAYS = days_range
    df = get_data()[['1stDet', 'BAND_r', 'BAND_i', 'BAND_g', 'MJD', 'SNID']]
    df = average_bands(df)
    return df

if __name__ == '__main__':
    df = run_pipeline(2)
    df.to_csv(f'./out/output_{NUM_DAYS}.csv') 
# print('Unique SNID 2 day: ', df['SNID'].nunique())

# take average of days

# plt.scatter(x = df['BAND_r']-df['BAND_i'], y )