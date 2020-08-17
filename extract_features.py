from instance import Instance
from mip.callbacks import CutPool
from mip.model import compute_features, features
from mip.entities import Constr
from mip.constants import INTEGER, BINARY, CONTINUOUS, OptimizationStatus
from mip.constants import CutType
import time
import inspect
import os


class ExtractFeatures:
    def __init__(self, instance: Instance):
        self.instance = instance
        if os.path.exists('{}_dataset.csv'.format(self.instance.name)):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        self.csv = open('{}_dataset.csv'.format(self.instance.name), append_write)
        if os.path.exists('{}_dataset.log'.format(self.instance.name)):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        self.log = open('{}_dataset.log'.format(self.instance.name), append_write)

        self.dataset = list()
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

        # decision label
        self.feat_names.append('label')

        self.test_ok = 0
        self.test_false = 0

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
        while qtd_cuts > 0:
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
                try:
                    cp = self.instance.model.generate_cuts([type_cut], 8192, 1e-4)
                    print(type_cut, len(cp.cuts))
                    self.log.write('{}: {}\n'.format(type_cut, len(cp.cuts)))
                    qtd_cuts = qtd_cuts + len(cp.cuts)
                    if len(cp.cuts) > 0:
                        # input()
                        for c in cp.cuts:
                            qtd = qtd + 1
                            self.instance.model += c, 'cut_{}({})'.format(type_cut.value, qtd)
                            feat_cut, label = self.label_cut(c, type_cut)
                            if label > 0:
                                self.test_ok = self.test_ok + 1
                            else:
                                self.test_false = self.test_false + 1
                            feat = self.features(iteration, type_cut, c, zeros, unsatis, label, feat_cut)
                            self.dataset.append(feat)
                except Exception as e:
                    # print(inspect.trace()[-1][0].f_locals)
                    type_cut = inspect.trace()[-1][0].f_locals['cut_types'][0]
                    print('exception {}'.format(e))
                    print('error in cut {}'.format(type_cut))
                    self.log.write('exception {}\n'.format(e))
                    self.log.write()
                    # cp = inspect.trace()[-1][0].f_locals['cp']
                    # print(len(cp.cuts))
                    # if len(cp.cuts) > 0:
                    #     c = inspect.trace()[-1][0].f_locals['cut']
                    #     print(c)
                    # input('erro')
                    pass

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
        self.csv.write('{}\n'.format(out))

        for d in self.dataset:
            out = str(d[0])
            for v in d[1:]:
                out = out + ';{}'.format(v)
            self.csv.write('{}\n'.format(out))
        self.csv.close()
        print('len dataset', len(self.dataset))
        print('test_ok', self.test_ok)
        print('test_false', self.test_false)

        self.log.write('len dataset: {}\n'.format(len(self.dataset)))
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
