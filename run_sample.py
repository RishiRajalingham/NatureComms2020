
import decoder_utils
import pickle as pk
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
start_time = time.time()

sample_data_path = '/om/user/rishir/lib/NeuronalRecycling/sample_data/' # <- EDIT THIS TO POINT TO CORRECT PATH
sample_data_fn_IT = '%s/data_IT_base616.pkl' % sample_data_path
sample_data_fn_bab = '%s/bab.pkl' % sample_data_path

IT_base616 = pk.load(open(sample_data_fn_IT, 'rb'))
features, meta = IT_base616['IT_features'], IT_base616['meta']
oc = decoder_utils.ObjectClassifier()
IT_rec = oc.get_classification_record(features, meta)

perf_mu = np.nanmean([x['all']['perf'].mean() for x in IT_rec])
perf_sig = np.nanstd([x['all']['perf'].mean() for x in IT_rec])
print('IT decoder performance: %2.2f +- %2.2f' % (perf_mu, perf_sig))

bc = decoder_utils.BehaviorCharacterizer()
met_IT = bc.get_metrics(IT_rec, meta)

bab_dat = pk.load(open(sample_data_fn_bab, 'rb'), encoding='latin1')
perf_mu = np.nanmean([x['all']['perf'].mean() for x in bab_dat])
perf_sig = np.nanstd([x['all']['perf'].mean() for x in bab_dat])
print('Baboon performance: %2.2f +- %2.2f' % (perf_mu, perf_sig))

bc = decoder_utils.BehaviorCharacterizer(resp_var='resp')
met_bab = bc.get_metrics(bab_dat, meta)
#
out = decoder_utils.get_consistency(met_IT, met_bab, metricn='grp5_bigram_freq_dprime')
rho_mu = np.nanmean(out['rho_n'])
rho_sig = np.nanstd(out['rho_n'])
print('IT-baboon consistency: %2.2f +- %2.2f' % (rho_mu, rho_sig))

# your code
elapsed_time = time.time() - start_time
tmp = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('Elapsed time: %s' % tmp)