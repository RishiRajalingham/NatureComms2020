import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scipy.stats import norm, pearsonr, spearmanr


class ObjectClassifier(object):
    def __init__(self, **kwargs):
        self.C = kwargs.get('C', 1.0)
        self.max_iter = kwargs.get('max_iter', 10000)
        self.decode_key = kwargs.get('decode_key', 'label_word')
        self.split_key = kwargs.get('split_key', 'id')
        self.k = kwargs.get('k', 10)
        self.niter = kwargs.get('niter', 5)
        self.niter_within_sample = kwargs.get('niter_within_sample', 1)
        self.nfeat_sample = kwargs.get('nfeat_sample', None)
        self.pca_preprocess = kwargs.get('pca_preprocess', True)
        self.run_splits = kwargs.get('run_splits', True)
        self.nreps = kwargs.get('nreps', None)

        self.clf = LogisticRegression(multi_class='multinomial',
                                      solver='newton-cg',
                                      C=self.C,
                                      max_iter=self.max_iter)
        return

    @staticmethod
    def get_train_test_splits(X, y, group, k=10):
        if group is None:
            kf = StratifiedKFold(k)
            kf.get_n_splits(X, y)
            splits = kf.split(X, y)
        else:
            kf = GroupKFold(k)
            kf.get_n_splits(X, y, group)
            splits = kf.split(X, y, group)
        return splits

    @staticmethod
    def get_finite_features(X_train, X_test):
        idx_no_var = (np.nanstd(X_train, axis=0) > 0) & \
                     (np.isfinite(np.mean(X_train, axis=0)))
        X_train = X_train[:, idx_no_var]
        X_test = X_test[:, idx_no_var]
        return X_train, X_test

    @staticmethod
    def format_feature_matrix(X, rep_axis=2):
        X = np.nanmean(X, axis=rep_axis)
        s = X.shape
        X_ = np.reshape(X, (s[0], s[1] * s[2]))
        idx = np.isfinite(np.mean(X_, axis=0))
        return X_[:, idx]

    @staticmethod
    def balance_classes(meta, decode_key='label_word'):
        idx0 = np.nonzero(meta[decode_key] == 0)[0]
        idx1 = np.nonzero(meta[decode_key] == 1)[0]
        if len(idx0) == len(idx1):
            idx = list(range(meta.shape[0]))
            return idx

        n = np.min([len(idx0), len(idx1)])
        idx0_ = np.random.choice(idx0, n, replace=False)
        idx1_ = np.random.choice(idx1, n, replace=False)
        idx = np.concatenate((idx0_, idx1_), axis=0)
        return idx

    @staticmethod
    def preprocess_features(X_train, X_test, pca_preprocess=False):
        if pca_preprocess:
            ncomp = int((min(X_train.shape) - 1))
            pca = PCA(n_components=ncomp, svd_solver='arpack')
            pca = pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
        else:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test

    def get_feature_sample(self, features, neural_meta=None):
        features_sample = {}
        fs = features.shape
        nim, nneurs, = fs[0], fs[1]

        # features.shape: im, neur, reps, timebins
        if self.nfeat_sample is not None:
            if (neural_meta is not None) and (self.nfeat_sample > 1):
                neur_ch_L = np.where(neural_meta == 'L')[0]
                neur_ch_R = np.where(neural_meta == 'R')[0]
                i1 = int(self.nfeat_sample / 2.0)
                i2 = self.nfeat_sample - i1
                neur_idx_L = np.random.choice(neur_ch_L, i1, replace=False)
                neur_idx_R = np.random.choice(neur_ch_R, i2, replace=False)
                neur_idx = np.concatenate((neur_idx_L, neur_idx_R), axis=0)
            else:
                nneur_ch = range(nneurs)
                neur_idx = np.random.choice(nneur_ch, self.nfeat_sample, replace=False)
            features = features[:, neur_idx, ...]

        if len(fs) > 2:
            if self.nreps is None:
                self.nreps = fs[2]

            treps = list(range(fs[2]))
            np.random.shuffle(treps)

            rep_idx = np.random.choice(treps, self.nreps, replace=False)

            sh1 = rep_idx[:np.int(self.nreps / 2)]
            sh2 = rep_idx[np.int(self.nreps / 2):]

            if len(fs) == 4:
                features_sample['split1'] = self.format_feature_matrix(features[:, :, sh1, :])
                features_sample['split2'] = self.format_feature_matrix(features[:, :, sh2, :])
                features_sample['all'] = self.format_feature_matrix(features)

            elif len(fs) == 3:
                # im x neur x reps
                features_sample['split1'] = np.nanmean(features[:, :, sh1], axis=2)
                features_sample['split2'] = np.nanmean(features[:, :, sh2], axis=2)
                features_sample['all'] = np.nanmean(features, axis=2)
        else:
            features_sample['split1'], features_sample['split2'], features_sample['all'] = features, features, features

        return features_sample

    def classify_base(self, features, meta):
        recs_over_iter = []
        for i in range(self.niter_within_sample):

            X = features
            y = np.array(meta[self.decode_key])
            if (self.split_key is not None) and (self.split_key != 'none'):
                g = np.array(meta[self.split_key])
            else:
                g = None
            splits = self.get_train_test_splits(X, y, g, self.k)

            ypred = np.ones(y.shape) * np.nan
            yprob = np.ones(y.shape) * np.nan
            yperf = np.ones(y.shape) * np.nan

            for tr_index, te_index in splits:
                X_train, X_test = X[tr_index], X[te_index]
                y_train, y_test = y[tr_index], y[te_index]
                X_train, X_test = self.preprocess_features(X_train, X_test, self.pca_preprocess)
                X_train, X_test = self.get_finite_features(X_train, X_test)

                clf_curr = clone(self.clf)
                clf_curr = clf_curr.fit(X_train, y_train)
                score = clf_curr.predict(X_test)
                proba = clf_curr.predict_proba(X_test)

                cls_order = clf_curr.classes_
                positive_index = np.nonzero(cls_order == 1)[0][0]
                ypred[te_index] = score
                yperf[te_index] = score == y[te_index]
                yprob[te_index] = proba[:, positive_index]

            yprob_discrete = (yprob > 0.5).astype(float)
            rec_curr = pd.DataFrame({
                'imgdata': meta['id'], 'word': meta['word'],
                'resp_prob': yprob, 'perf': yperf,
                'resp_disc': yprob_discrete})
            recs_over_iter.append(rec_curr)

        rec = pd.concat(recs_over_iter)
        rec = rec.reset_index(drop=True)

        return rec

    def get_classification_record(self, features, meta, neural_meta=None):
        rec_features = []

        if self.run_splits:
            split_names = ['all', 'split1', 'split2']
        else:
            split_names = ['all']

        for i in range(self.niter):
            features_sample = self.get_feature_sample(features, neural_meta=neural_meta)
            idx_ = self.balance_classes(meta, decode_key=self.decode_key)
            rec_sample = {}
            for fnk in split_names:
                f, m = features_sample[fnk][idx_, :], meta.iloc[idx_, :]
                rec_sample[fnk] = self.classify_base(f, m)
            rec_features.append(rec_sample)

        return rec_features


class BehaviorCharacterizer(object):
    def __init__(self, **kwargs):
        self.label_var = kwargs.get('label_var', 'label_word')
        self.resp_var = kwargs.get('resp_var', 'resp_prob')
        self.grpvars = kwargs.get('grpvars', ['all', 'grp5_bigram_freq'])
        return

    @staticmethod
    def get_dprime(hr, cr):
        return np.clip((norm.ppf(hr) - norm.ppf(1 - cr)), -5, 5)

    @staticmethod
    def get_balanced_accuracy(hr, cr):
        return (hr + cr) / 2

    @staticmethod
    def register_meta(data, meta, grpvars_):
        meta_id = list(meta['id'])
        ind = data['imgdata'].isin(meta['id'])
        curr_data = data[ind].reset_index(drop=True)
        grpvars = [li for li in grpvars_ if li not in curr_data.keys()]
        for gv in grpvars:
            curr_data[gv] = np.nan
        meta_ind = [meta_id.index(curr_data['imgdata'][i]) for i in curr_data.index]
        curr_data[grpvars] = meta.iloc[meta_ind, :][grpvars].reset_index(drop=True)
        return curr_data

    def get_metrics_base_withingrpvar(self, data, grpvar):
        resp1 = data.groupby([grpvar, self.label_var])[self.resp_var].mean()
        resp1 = resp1.unstack()

        hr0, cr0 = 1 - resp1[0], np.nanmean(resp1[1])
        hr1, cr1 = resp1[1], np.nanmean(1 - resp1[0])
        hr = np.concatenate((hr0, hr1))

        dp0 = self.get_dprime(hr0, cr0)
        dp1 = self.get_dprime(hr1, cr1)
        dp = np.concatenate((dp0, dp1))

        ba0 = self.get_balanced_accuracy(hr0, cr0)
        ba1 = self.get_balanced_accuracy(hr1, cr1)
        ba = np.concatenate((ba0, ba1))

        res = {'hitrate0': hr0, 'hitrate1': hr1, 'hitrate': hr,
               'dprime0': dp0, 'dprime1': dp1, 'dprime': dp,
               'balacc0': ba0, 'balacc1': ba1, 'balacc': ba}
        return res

    def get_metrics_base(self, data, meta):
        if 'all' not in meta.keys():
            meta['all'] = 1

        register_var = self.grpvars + [self.label_var]
        data = self.register_meta(data, meta, register_var)
        data[self.label_var] = data[self.label_var] > 0
        rec = {}
        for grpvar in self.grpvars:
            if grpvar == self.label_var:
                continue
            try:
                res = self.get_metrics_base_withingrpvar(data, grpvar)
            except:
                continue
            for r_fk in res.keys():
                rec['%s_%s' % (grpvar, r_fk)] = res[r_fk]

        """ We 	use classifier probability outputs for estimating metrics
        when computing consistencies (increased SNR), but use class choice output
        for computing performance (actual performance of classifier).
        """

        return rec

    def get_metrics(self, data, meta):
        recs = []
        for dat in data:
            rec = {}
            for fnk in dat.keys():
                rec[fnk] = self.get_metrics_base(dat[fnk], meta)
            recs.append(rec)
        return recs


""" consistency analyses """


def nnan_corr(A, B, corrtype='pearson'):
    ind = np.isfinite(A) & np.isfinite(B)
    A, B = A[ind], B[ind]
    if corrtype == 'pearson':
        return pearsonr(A, B)[0]
    elif corrtype == 'spearman':
        return spearmanr(A, B)[0]


def get_consistency_base(A1, A2, B1, B2, corrtype='pearson'):
    try:
        ic_a = nnan_corr(A1, A2, corrtype=corrtype)
        ic_b = nnan_corr(B1, B2, corrtype=corrtype)
        # restrict to across split folds, to ensure independence of data,
        # e.g. cross-validated regressions
        rho = np.nanmean([nnan_corr(A1, B2, corrtype=corrtype),
                          nnan_corr(A2, B1, corrtype=corrtype)])

        rho_n = rho / ((ic_a * ic_b) ** 0.5)
    except:
        ic_a = ic_b = rho = rho_n = np.nan
    res = {'IC_a': ic_a, 'IC_b': ic_b,
           'rho': rho, 'rho_n': rho_n}
    return res


def get_consistency(A, B, metricn='grp5_signed_dprime', corrtype='pearson'):
    out = {'IC_a': [], 'IC_b': [],
           'rho': [], 'rho_n': []}

    for dat_a in A:
        a, a1, a2 = dat_a['all'][metricn], dat_a['split1'][metricn], dat_a['split2'][metricn]
        for dat_b in B:
            b, b1, b2 = dat_b['all'][metricn], dat_b['split1'][metricn], dat_b['split2'][metricn]

            nnan_idx = np.isfinite(a1) & np.isfinite(a2) \
                       & np.isfinite(b1) & np.isfinite(b2)
            A1, A2 = a1[nnan_idx], a2[nnan_idx]
            B1, B2 = b1[nnan_idx], b2[nnan_idx]
            out_curr = get_consistency_base(A1, A2, B1, B2, corrtype=corrtype)
            for o_fk in out_curr.keys():
                out[o_fk].append(out_curr[o_fk])

    return out
