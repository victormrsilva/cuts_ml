from instance import Instance
from mip.callbacks import CutPool
from mip.model import compute_features, features
from mip.entities import Constr, LinExpr
from mip.constants import INTEGER, BINARY, CONTINUOUS, OptimizationStatus
from mip.constants import CutType
import time
import inspect
import os
import random
import pandas as pd


class ExtractFeatures:
    def __init__(self, instance: Instance):
        self.instance = instance
        if os.path.exists('{}_dataset.log'.format(self.instance.name)):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        self.log = open('{}_dataset.log'.format(self.instance.name), append_write)

        self.dataset = [[[[] for i in range(2)] for j in range(15)] for k in range(10)]
        self.feat_names = features()
        self.instance_feat_values = compute_features(self.instance.model)

        self.feat_names.append('instance')

        # features da relaxação
        self.feat_names.append('relax_iteration')
        self.feat_names.append('nonzeros')  # ###
        self.feat_names.append('pct_nonzeros')  # ###
        self.feat_names.append('unsatisfied_var')  # variavel inteira que ainda está como fract
        self.feat_names.append('pct_unsatisfied_var')  # variavel inteira que ainda está como fract

        # features do corte
        self.feat_names.append('cut_type')
        self.feat_names.append('n_variables_coef_nonzero')
        self.feat_names.append('xvar_zero')  # qtd vars in cut equal zero
        self.feat_names.append('coeff_leq_0.5')  # qtd vars in cut equal zero
        self.feat_names.append('coeff_leq_1')  # qtd vars in cut equal zero
        self.feat_names.append('minor_absolute_coef')  # ###
        self.feat_names.append('abs_minor_absolute_coef')  # ###
        self.feat_names.append('major_absolute_coef')  # ###
        self.feat_names.append('abs_major_absolute_coef')  # ###
        self.feat_names.append('abs_rhs')  # ###
        self.feat_names.append('abs_ratio_minor_major_coef')  # ###
        self.feat_names.append('rhs')  # ###
        self.feat_names.append('sense')  # ###
        self.feat_names.append('diff')  # ###
        self.feat_names.append('away')  # ###
        self.feat_names.append('lub')  # ###
        self.feat_names.append('eps_coeff')  # ###
        self.feat_names.append('eps_coeff_lub')  # ###


        # decision label
        self.feat_names.append('label')

        self.test_ok = 0
        self.test_false = 0

    def features_gerard(self, c: LinExpr):
        feat = dict()
        feat['away'] = 0
        feat['lub'] = 0
        feat['eps_coeff'] = 0
        feat['eps_coeff_lub'] = 0

        AWAY = 1e-2
        EPS_COEFF = 1e-11

        for var, coef in c.expr.items():
            # print(var.x, coef, var.ub, var.lb, AWAY, abs(var.x - var.ub) < AWAY, abs(var.x - var.lb) < AWAY)
            if abs(var.x - round(var.x)) < AWAY:
                feat['away'] = feat['away'] + 1

            if abs(var.x - var.ub) < AWAY or abs(var.x - var.lb) < AWAY:
                feat['lub'] = feat['lub'] + 1
                if abs(coef) < EPS_COEFF:
                    feat['eps_coeff_lub'] = feat['eps_coeff_lub'] + 1
            else:
                if abs(coef) < EPS_COEFF:
                    feat['eps_coeff'] = feat['eps_coeff'] + 1
        return feat

    def zeros_unsatis(self):
        zeros = 0
        unsatis = 0
        for x in self.instance.model.vars:
            if x.x > self.instance.model.infeas_tol:
                zeros = zeros + 1
            if (x.var_type == INTEGER or x.var_type == BINARY) and abs(x.x - (int(x.x)) > self.instance.model.infeas_tol):
                unsatis = unsatis + 1
            # print(x.x, int(x.x), abs(x.x - int(x.x)), x.var_type, zeros, unsatis)
            # input()
        return zeros, unsatis

    def features(self, iteration, type_cut, c, nzeros, unsatis, label, feat_cut):
        feat = self.instance_feat_values.copy()

        feat.append(self.instance.name)

        # relaxation features
        feat.append(iteration)
        feat.append(nzeros)
        feat.append(round(nzeros/self.instance.model.num_cols, 6))
        feat.append(unsatis)
        feat.append(round(unsatis / self.instance.model.num_int, 6))

        # cut features
        min_coef = min(c.expr.items(), key=lambda x: x[1])[1]
        max_coef = max(c.expr.items(), key=lambda x: x[1])[1]
        feat.append(type_cut.name)
        feat.append(len(c.expr))
        feat.append(feat_cut['xvar_zero'])
        feat.append(feat_cut['coeff_leq_0.5'])
        feat.append(feat_cut['coeff_leq_1'])
        feat.append(min_coef)
        feat.append(abs(min_coef))
        feat.append(max_coef)
        feat.append(abs(max_coef))
        feat.append(abs(min_coef/max_coef))
        feat.append(c.const)
        feat.append(abs(c.const))
        feat.append(c.sense)
        feat.append(feat_cut['diff'])

        new_feat = self.features_gerard(c)
        feat.append(new_feat['away'])
        feat.append(new_feat['lub'])
        feat.append(new_feat['eps_coeff'])
        feat.append(new_feat['eps_coeff_lub'])

        # label
        feat.append(label)  # label

        return feat

    def test(self, combinatory: [CutType], noncombinatory: [CutType]):

        self.log.write('Combinatory set: {}\n'.format(combinatory))
        self.log.write('Non-Combinatory set: {}\n'.format(noncombinatory))

        qtd = 0
        iteration = 0
        qtd_cuts = 1
        # temporizar as rodadas para ver se é necessário a limitação de cortes inseridos
        while qtd_cuts > 0 and iteration < 10:
            self.log.write('iteration {}\n'.format(iteration))
            print('iteration {}'.format(iteration))
            qtd_cuts = 0
            start = time.time()
            self.instance.model.optimize(relax=True)
            end = time.time()
            print('relaxation time {} seconds'.format(round(end-start, 4)))
            self.log.write('relaxation time {} seconds\n'.format(round(end-start, 4)))

            # checar solução do corte
            # se não for ótima há algum problema com algum corte
            if self.instance.model.status != OptimizationStatus.OPTIMAL:
                self.log.write('relaxação se tornou infeasible\n')
                print('relaxação se tornou infeasible')
                qtd_cuts = 0
                continue

            zeros, unsatis = self.zeros_unsatis()

            cp = self.instance.model.generate_cuts(combinatory, 8192, 1e-4)
            qtd_cuts = qtd_cuts + len(cp.cuts)
            self.log.write('CutType.COMBINATORY: {}\n'.format(len(cp.cuts)))
            print('CutType.COMBINATORY: {}'.format(len(cp.cuts)))
            if len(cp.cuts) > 0:
                # input()
                for c in cp.cuts:
                    qtd = qtd + 1
                    self.instance.model += c, 'cut_combinatory({})'.format(qtd)

            for type_cut in noncombinatory:
                # try:
                    cp = self.instance.model.generate_cuts([type_cut], 8192, 1e-4)
                    print(type_cut, len(cp.cuts))
                    self.log.write('{}: {}\n'.format(type_cut, len(cp.cuts)))

                    if len(cp.cuts) > 0:
                        i = 0
                        for c in cp.cuts:
                            if i < 30:
                                feat_cut, label = self.label_cut(c, type_cut)
                                if label > 0:
                                    qtd = qtd + 1
                                    self.test_ok = self.test_ok + 1
                                    i = i + 1
                                    qtd_cuts = qtd_cuts + 1
                                    self.instance.model += c, 'cut_{}({})'.format(type_cut.value, qtd)
                                else:
                                    self.test_false = self.test_false + 1
                                feat = self.features(iteration, type_cut, c, zeros, unsatis, label, feat_cut)
                                if len(self.dataset[iteration][type_cut.value][label]) == 10:
                                    self.dataset[iteration][type_cut.value][label].pop(random.randint(0, 9))
                                self.dataset[iteration][type_cut.value][label].append(feat)
                # except Exception as e:
                    # print(inspect.trace()[-1][0].f_locals)
                    # type_cut = '' # inspect.trace()[-1][0].f_locals['cut_types'][0]
                    # print('exception {}'.format(e))
                    # print('error in cut {}'.format(type_cut))
                    # self.log.write('exception {}\n'.format(e))
                    # self.log.write('error in cut {}'.format(type_cut))
                    # cp = inspect.trace()[-1][0].f_locals['cp']
                    # print(len(cp.cuts))
                    # if len(cp.cuts) > 0:
                    #     c = inspect.trace()[-1][0].f_locals['cut']
                    #     print(c)
                    # input('erro')
                    # pass

            # add noncombinatory
            # print(noncombinatory)

            # print(generated_cuts_per_iteration)
            # print(total_cuts)
            self.log.flush()
            iteration = iteration + 1
        # input()
        self.writeEnd()

    def writeEnd(self):
        # self.instance.model.write('{}.lp'.format(self.instance.name))
        out = str(self.feat_names[0])
        for v in self.feat_names[1:]:
            out = out + ';{}'.format(v)

        for i in range(10):
            for t in range(15):
                for label in range(2):
                    if len(self.dataset[i][t][label]) > 0:
                        filename = '{}_{}_{}_{}_dataset.csv'.format(self.instance.name, i, t, label)
                        if not os.path.exists('{}_{}_{}_{}_dataset.csv'.format(self.instance.name, i, t, label)):
                            file = open(filename, 'w')
                            file.write('{}\n'.format(out))
                            file.close()
                        csv_file = pd.read_csv(filename, delimiter=';')
                        for feat in self.dataset[i][t][label]:
                            while len(csv_file) >= 10:
                                # print(t)
                                # print(csv_file)
                                csv_file = csv_file.drop(index=random.randint(0, len(csv_file)-1))
                                # input(csv_file)

                            s = pd.Series(feat, index=csv_file.columns)
                            if s.values.tolist() not in csv_file.values.tolist():
                                csv_file = csv_file.append(s, ignore_index=True)
                        csv_file.to_csv(filename, sep=';', index=False)

        print('test_ok', self.test_ok)
        print('test_false', self.test_false)

        self.log.write('test_ok: {}\n'.format(self.test_ok))
        self.log.write('test_false: {}\n'.format(self.test_false))
        self.log.close()

    def label_cut(self, c: Constr, type: CutType):
        feat_cut = dict()
        feat_cut['xvar_zero'] = 0  # qtd vars in cut equal zero
        feat_cut['coeff_leq_0.5'] = 0  # qtd vars in cut equal zero
        feat_cut['coeff_leq_1'] = 0  # qtd vars in cut equal zero

        for key, value in c.expr.items():
            if abs(key.x) < self.instance.model.infeas_tol:
                # print(key, key.x)
                feat_cut['xvar_zero'] = feat_cut['xvar_zero'] + 1
            # else :
            #     print('----', key, key.x)
            if abs(value) <= (0.5 + self.instance.model.infeas_tol):
                # print(abs(value), '******')
                feat_cut['coeff_leq_0.5'] = feat_cut['coeff_leq_0.5'] + 1
            elif (0.5 + self.instance.model.infeas_tol) < abs(value) < (1 + self.instance.model.infeas_tol):
                # print(abs(value), '........')
                feat_cut['coeff_leq_1'] = feat_cut['coeff_leq_1'] + 1
        # print(len(c.expr), feat_cut['xvar_zero'])
        # print(feat_cut['coeff_leq_0.5'], feat_cut['coeff_leq_1'])
        # input()

        for sol in self.instance.solutions:
            total = 0
            for key, value in c.expr.items():
                total = total + sol[key.name]*value
            # print(total, c.sense, (-1) * c.const)
            feat_cut['diff'] = total + c.const
            if c.sense == '<':
                if total + c.const > self.instance.model.infeas_tol:
                    print('checking constr', c)
                    # for key, value in c.expr.items():
                    #     print(key, value, sol[key.name])
                    print('type {}: {} {} {} - {}\n'.format(type, total, c.sense, (-1) * c.const, total - (-1) * c.const))
                    self.log.write('type {}: {} {} {} - {}\n'.format(type, total, c.sense, (-1) * c.const, total - (-1) * c.const))
                    # input(0)
                    return feat_cut, 0
            elif c.sense == '>':
                if total + c.const < - self.instance.model.infeas_tol:
                    print('checking constr', c)
                    # for key, value in c.expr.items():
                    #     print(key, value, sol[key.name])
                    print('type {}: {} {} {} - {}\n'.format(type, total, c.sense, (-1) * c.const, total - (-1) * c.const))
                    self.log.write('type {}: {} {} {} - {}\n'.format(type, total, c.sense, (-1) * c.const, total - (-1) * c.const))
                    # input(0)
                    return feat_cut, 0

        return feat_cut, 1
