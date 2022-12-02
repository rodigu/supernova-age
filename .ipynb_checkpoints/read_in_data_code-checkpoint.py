#!/usr/bin/env python3.10
def plot_snana_fits(dir_path, plotmag=False):
    # dir_path is filepath to model files
    import sncosmo
    import os 
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas
    
    heads = sorted(glob.glob(os.path.join(dir_path, '*_HEAD.FITS.gz')))
    phots = sorted(glob.glob(os.path.join(dir_path, '*_PHOT.FITS.gz')))
    assert len(heads) != 0, 'no *_HEAD_FITS.gz are found'
    assert len(heads) == len(phots), 'there are different number of HEAD and PHOT files'
    appended_data = []
    appended_meta_data = []
    for head, phot in zip(heads[0:3], phots[0:3]): #lots of LCs per head, phot files, so do a few to start
        i = head.find('_HEAD.FITS.gz')
        assert head[:i] == phot[:i], f'HEAD and PHOT files name mismatch: {head}, {phot}'
        filename = head[:i].split('/')[-1].split('.')[0]
        for LCnum, lc in enumerate(sncosmo.read_snana_fits(head, phot)[0:10]): # remember: multiple SN in single HEAD/PHOT file
            # print(lc.meta)
            #print(lc.columns)
            lc_df = lc.to_pandas()
            lc_df_meta = pandas.DataFrame(lc.meta, columns=lc.meta.keys())
        
            # fig, ax = plt.subplots()
            lc_df['BAND']= lc_df['BAND'].str.decode("utf-8") # turn Bytes into str
            lc_df['SNR'] = lc_df['FLUXCAL']/lc_df['FLUXCALERR'] # signal to noise ratio
            lc_df['MAG'] = np.array(-2.5*np.log10(np.abs(lc_df['FLUXCAL'])))+27.5 # magnitudes
            lc_df['MAGERR'] = 1.086/lc_df['SNR'] # magnitude error
            
                      # observations              # nondetections           # Signal to noise cut       
            mask = (lc_df['PHOTFLAG'] != 0)  #| (lc_df['PHOTFLAG'] == 0)  | (lc_df['SNR'] >= 4) 
            lc_df = lc_df[mask].reset_index(drop=True)
            
            D_id_color = {
                      "X ": u"#b9ac70", # ZTF-g
                      "Y ": u"#bd1f01", # ZTF-r
                      "g ": u"#4daf4a", # YSE-g
                      "r ": u"#e41a1c", # YSE-r
                      "i ": u"#832db6", # YSE-i
                      "z ": u"#656364"} # YSE-z
            lc_df['PLOTCOLOR'] = lc_df.BAND.map(D_id_color)
            appended_meta_data.append(lc_df_meta)
            for pb, c in D_id_color.items():
                #print(pb, c)
                lc_df_pb = lc_df[lc_df.BAND == pb]
                appended_data.append(lc_df_pb)

                #print(lc_df_pb.ZEROPT)
            
#                 if plotmag:
#                     plt.errorbar(lc_df_pb['MJD']-lc.meta['MJD_TRIGGER'], lc_df_pb['MAG'], 
#                              yerr=lc_df_pb['MAGERR'], c=c, fmt='o', label=pb, ms=7, elinewidth=2)
                    
#                 else:
#                     plt.errorbar(lc_df_pb['MJD']-lc.meta['MJD_TRIGGER'], lc_df_pb['FLUXCAL'], 
#                              yerr=lc_df_pb['FLUXCALERR'], c=c, fmt='o', label=pb, ms=7, elinewidth=2)
                
                
#             if plotmag:
#                 plt.gca().invert_yaxis()
#                 plt.ylabel('Mag')
                
#             else:
#                 plt.ylabel('Flux')
                
#             plt.show()
    appended_data = pandas.concat(appended_data)     
    return lc_df_meta