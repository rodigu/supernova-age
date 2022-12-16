from read_in_data_code import plot_snana_fits
import matplotlib.pyplot as plt
import sncosmo
import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas
import time

def get_data(save_raw=False, days_range=1):
    models = os.listdir('data/')[1::]
    heads_list = []
    phots_list = []
    print("Reading models")
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

            
    D_id_color = {
            "X ": u"#b9ac70", # ZTF-g
            "Y ": u"#bd1f01", # ZTF-r
            "g ": u"#4daf4a", # YSE-g
            "r ": u"#e41a1c", # YSE-r
            "i ": u"#832db6", # YSE-i
            "z ": u"#656364"} # YSE-z

    # skip_size = 100
    for head, phot in zip(heads_list, phots_list): #lots of LCs per head, phot files, so do a few to start
        i = head.find('_HEAD.FITS.gz')
        assert head[:i] == phot[:i], f'HEAD and PHOT files name mismatch: {head}, {phot}'
        filename = head[:i].split('/')[1:3]#.split('.')[0:2]
        # num_heads = 50
        print('Current file: ', filename)
        start_time = time.time()
        for LCnum, lc in enumerate(sncosmo.read_snana_fits(head, phot)): # remember: multiple SN in single HEAD/PHOT file
            lc_meta = {lc.meta['SNID']:lc.meta}
            
            #print(lc.columns)
            lc_df = lc.to_pandas()
            #               {'01': 'SNIa', '12': 'SNII', '20': 'SNII', '33': 'SNII', '37': 'SNIbc’}
            model_to_type = {'01': 0, '12': 1, '20': 1, '33': 1, '37': 2}
            # lc_df_meta = pandas.DataFrame.from_dict(lc_meta,orient='columns')
            lc_df['parsnip_type'] = model_to_type[filename[0][-2:]]
            lc_df['SNID']=lc.meta['SNID']
            lc_df['1stDet']=lc.meta['MJD_DETECT_FIRST']


            # lc_df['Trigger']=lc.meta['MJD_TRIGGER']
            # lc_df['Model_num']=lc.meta['SIM_TYPE_INDEX']
            # lc_df['filename']=filename[1]
            
            # fig, ax = plt.subplots()
            lc_df['BAND']= lc_df['BAND'].str.decode("utf-8") # turn Bytes into str
            lc_df['SNR'] = lc_df['FLUXCAL']/lc_df['FLUXCALERR'] # signal to noise ratio
            lc_df['MAG'] = np.array(-2.5*np.log10(np.abs(lc_df['FLUXCAL'])))+27.5 # magnitudes
            lc_df['MAGERR'] = 1.086/lc_df['SNR'] # magnitude error
            
                    # observations              # nondetections           # Signal to noise cut       
            mask = ((lc_df['PHOTFLAG'] != 0) | (lc_df['SNR'] >= 4))  #| (lc_df['PHOTFLAG'] == 0)  | (lc_df['SNR'] >= 4) 
            lc_df = lc_df[mask].reset_index(drop=True)
            # lc_df['PLOTCOLOR'] = lc_df.BAND.map(D_id_color)

            # dropping supernovas older than 30 days
            lc_df = lc_df[lc_df['MJD'] - lc_df['1stDet'] < 30]
            
            if type(appended_data) == int:
                appended_data = lc_df.copy()
            else:
                appended_data = pandas.concat([appended_data, lc_df])
        
        time_taken = time.time() - start_time
        print((f"   Time taken: {time_taken} seconds"))
        print("-----")
    
    return masked_data(appended_data, save_raw, days_range)

def masked_data(df, save_raw=False, days_range=1):
    df['int_MJD'] = df['MJD'].astype(int)
    df['norm_MJD'] = (df['MJD'] / days_range).astype(int)
    for b in BAND_CHOICE:
        df[b] = df.apply(lambda row: np.NaN if row['BAND'] != f'{b[-1]} ' else row['MAG'],axis=1)
    return df
    # print('Unique SNID sample: ', df['SNID'].nunique())
    lc_df_int_MJD = df.groupby(['SNID', 'norm_MJD']).mean(['BAND_r','BAND_i','BAND_g']).reset_index()
    band_mask = (~lc_df_int_MJD['BAND_r'].isna()) & (~lc_df_int_MJD['BAND_g'].isna()) & (~lc_df_int_MJD['BAND_i'].isna())
    return df[df['norm_MJD'].isin(lc_df_int_MJD[band_mask]['norm_MJD'])]

def average_bands(df):
    snid_df = df.groupby(['SNID', 'norm_MJD']).mean(numeric_only=True).reset_index().set_index('SNID')
    return snid_df.dropna()

def run_pipeline(days_range=1, save_raw=False):
    df = get_data(save_raw, days_range)
    adf = average_bands(df[['1stDet', 'MJD', 'norm_MJD', 'SNID', 'parsnip_type', *BAND_CHOICE]])
    adf['parsnip_type'] = adf['parsnip_type'].astype(int)
    print(f'Usable data: {len(adf) / len(df): .3}')
    print(f'Total usable: {len(adf)}, out of {len(df)}')
    return adf

BAND_CHOICE = ['BAND_r', 'BAND_i', 'BAND_g']
DAY_RANGE = 1

if __name__ == '__main__':
    print('Running')
    day_range = DAY_RANGE

    start_time = time.time()
    df = run_pipeline(day_range)
    print("________________________")
    print(f'Day range: {DAY_RANGE}')
    print(f'Bands: {BAND_CHOICE}')
    print(f"    Total time taken: {time.time() - start_time} seconds")

    start_time = time.time()
    df.to_csv(f'./out/output_{day_range}_typed.csv')
    print(f"    Time to write file: {time.time() - start_time}")
# print('Unique SNID 2 day: ', df['SNID'].nunique())

# take average of days

# plt.scatter(x = df['BAND_r']-df['BAND_i'], y )