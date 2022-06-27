import os
import time
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection


class TAS:

    def __init__(self, file_dss, file_dti, dti_cutoff=10000, permutation_round=10000, rnd_seed=0):
        """
        Read DSS, DTI files, csv format

        DSS csv file: 1st column is the drug id, rest columns are the DSS in different cells
        DTI csv file: 1st column is the drug id, 1st row is the target id.
        """
        if isinstance(file_dss, pd.DataFrame) and isinstance(file_dti, pd.DataFrame):
            self.dss = file_dss
            self.dti = file_dti
        else:
            self.dss = pd.read_csv(file_dss, index_col=0)
            self.dti = pd.read_csv(file_dti, index_col=0)

        self.permutation_round = permutation_round
        self.cut_off = dti_cutoff
        self.seed = rnd_seed
        self.permutation_results = {}
        self.tas_real = {}
        self.rank_list = []
        self.rank_all = {}
        self.tas_df = pd.DataFrame()
        self.tas_pv = pd.DataFrame()
        self.tas_pa = pd.DataFrame()
        self.tas_permut_df = None
        self.tas_adjust = None

        self.dss_has_dti = self.dss[self.dss.index.isin(self.dti.index)]
        self.dss_has_dti = self.dss_has_dti.loc[self.dti.index]
        self.dss_np = np.array(self.dss_has_dti.values, dtype=np.float32)
        self.dti_np = np.array(self.dti.values, dtype=np.float32)

        self.cells = list(self.dss_has_dti.columns)
        self.drugIDs = list(self.dti.index)
        self.proteins = list(self.dti.columns)

    @staticmethod
    def tas_mean(dss_selected):
        # selected dss values of a protein in all cell lines; np.array.
        return dss_selected.mean(axis=0)

    def cal_tas_pvalue(self, ):
        # Permutation test
        start_time = time.time()
        for prot in self.proteins:
            self.permutation_results[prot] = []

        rg = np.random.default_rng(self.seed)

        for i in range(len(self.proteins)):
            for j in range(self.permutation_round):
                col_p = rg.permutation(self.dti_np[:, i])
                idx = np.where(col_p < self.cut_off)[0]
                self.permutation_results[self.proteins[i]].append(self.tas_mean(self.dss_np[idx,]))

        for key, value in self.permutation_results.items():
            self.permutation_results[key] = np.array(value)
        end_time = time.time()
        print('Done permutation, time used in min: ', (end_time - start_time) / 60)
        # Calculate and rank real TAS values; save in txt file targets with p-value <= 0.05
        for i in range(len(self.proteins)):
            col = self.dti_np[:, i]
            idx = np.where(col < self.cut_off)[0]
            self.tas_real[self.proteins[i]] = self.tas_mean(self.dss_np[idx,])
        # print('r_tas proteins: ', len(self.tas_real.keys()))

        for prot, value in self.tas_real.items():
            self.rank_all[prot] = {}
            for i in range(len(value)):
                cell = self.cells[i]
                permutated_tas = self.permutation_results[prot][:, i]
                real_tas = value[i]
                tas_rank = np.count_nonzero(permutated_tas >= real_tas) / len(permutated_tas)
                self.rank_all[prot][cell] = [real_tas, tas_rank]
                self.rank_list.append([prot, cell, real_tas, tas_rank])
                self.tas_df.loc[prot, cell] = real_tas
                self.tas_pv.loc[prot, cell] = tas_rank

        self.tas_permut_df = pd.DataFrame(self.rank_list, columns=['Target', 'Cell_line', 'TAS', 'PVal'])
        tas_all = []
        for cell in self.tas_permut_df.Cell_line.unique():
            tas_cell = self.tas_permut_df[self.tas_permut_df.Cell_line == cell].copy()
            tas_cell['PVAdjust'] = fdrcorrection(tas_cell.PVal)[1]
            tas_all.append(tas_cell)
        self.tas_adjust = pd.concat(tas_all)
        self.tas_pa = pd.pivot_table(self.tas_adjust, index='Target', columns='Cell_line', values='PVAdjust')

        return self.dss_has_dti, self.tas_adjust, self.tas_df, self.tas_pv, self.tas_pa

    def save_files(self, out_dir, fname):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.dss_has_dti.to_csv(out_dir + 'DSS_has_DTI_%s.csv' % fname)
        self.tas_adjust.to_csv(out_dir + 'tas_permutation_padjusted_%s.csv' % fname)
        self.tas_df.to_csv(out_dir + 'tas_df_%s.csv' % fname)
        self.tas_pv.to_csv(out_dir + 'tas_pvalue_%s.csv' % fname)
        self.tas_pa.to_csv(out_dir + 'tas_adjusted_pvalue_%s.csv' % fname)
