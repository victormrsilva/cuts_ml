from instance import Instance
from mip.callbacks import CutPool
from mip.model import compute_features, features
from mip.entities import Constr
from mip.constants import INTEGER, BINARY, CONTINUOUS, OptimizationStatus
from mip.constants import CutType
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
        # features da relaxação
        self.feat_names.append('relax_iteration')
        self.feat_names.append('nonzeros')  # ###
        self.feat_names.append('pct_nonzeros')  # ###
        self.feat_names.append('unsatisfied_var')  # variavel inteira que ainda está como fract
        self.feat_names.append('pct_unsatisfied_var')  # variavel inteira que ainda está como fract

        # features do corte
        self.feat_names.append('cut_type')
        self.feat_names.append('n_variables_coef_nonzero')
        self.feat_names.append('abs_minor_absolute_coef')  # ###
        self.feat_names.append('minor_absolute_coef')  # ###
        self.feat_names.append('abs_major_absolute_coef')  # ###
        self.feat_names.append('major_absolute_coef')  # ###
        self.feat_names.append('abs_ratio_minor_major_coef')  # ###
        self.feat_names.append('abs_rhs')  # ###
        self.feat_names.append('rhs')  # ###

        # decision label
        self.feat_names.append('label')

        self.test_ok = 0
        self.test_false = 0

    def test(self, combinatory: [CutType], noncombinatory: [CutType]):

        # print(feat_values)
        # print(feat)

        generated_cuts_per_iteration = [[] for x in range(14)]  # 14 cuts in mip.constants
        total_cuts = [0 for x in range(14)]  # 14 cuts in mip.constants

        qtd = 0
        total_iteration = 0
        cuts_iteration = 1
        # temporizar as rodadas para ver se é necessário a limitação de cortes inseridos
        while cuts_iteration > 0:
            cuts_iteration = 0
            self.instance.model.optimize(relax=True)

            # checar solução do corte
            # se não for ótima há algum problema com algum corte
            if self.instance.model.status != OptimizationStatus.OPTIMAL:
                self.log.write('relaxação se tornou infeasible')
                print('relaxação se tornou infeasible')
                cuts_iteration = 0
                continue

            # print('generated_cuts_per_iteration', generated_cuts_per_iteration)
            # add combinatory cuts first
            # print(combinatory)
            for type_cut in combinatory+noncombinatory:
                try:
                    cp = self.instance.model.generate_cuts([type_cut], 8192, 1e-4)
                    print(type_cut, len(cp.cuts))
                    cuts_iteration = cuts_iteration + len(cp.cuts)
                    if len(cp.cuts) > 0:
                        total_cuts[type_cut.value] = total_cuts[type_cut.value] + len(cp.cuts)
                        print(type_cut, generated_cuts_per_iteration[type_cut.value])
                        # input()
                        for c in cp.cuts:
                            qtd = qtd + 1
                            self.instance.model += c, 'cut_{}({})'.format(type_cut.value, qtd)
                            cut_type_found = total_cuts[type_cut.value]
                            if type_cut in noncombinatory:
                                label = self.label_cut(c, type_cut)
                                if label > 0:
                                    test_ok = self.test_ok + 1
                                else:
                                    test_false = self.test_false + 1
                                feat = self.instance_feat_values.copy()
                                feat.append(type_cut.name)  # cut_type
                                feat.append(cut_type_found)  # cut_type_found
                                feat.append(total_iteration)  # iteration_found
                                feat.append(len(generated_cuts_per_iteration[type_cut.value]))  # cut_iteration
                                feat.append(len(c.expr.items()))  # n_variables
                                feat.append(label)  # label
                                self.dataset.append(feat)
                        generated_cuts_per_iteration[type_cut.value].append(len(cp.cuts))
                except Exception as e:
                    print(e.args)
                    # print(inspect.trace()[-1][0].f_locals)
                    type_cut = inspect.trace()[-1][0].f_locals['cut_types'][0]
                    print('error in cut', type_cut)
                    print(e.args)
                    self.log.write('error in cut {}\n'.format(type_cut))
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
            total_iteration = total_iteration + 1
        # input()
        self.writeEnd()
        self.instance.model.write('{}.lp'.format(self.instance.name))

    def writeEnd(self):
        out = str(self.feat_names[0])
        for v in self.feat_names[1:]:
            out = out + ';{}'.format(v)
        self.csv.write('{}\n'.format(str))

        for d in self.dataset:
            out = str(d[0])
            for v in d[1:]:
                out = out + ';{}'.format(v)
            self.csv.write('{}\n'.format(out))

        print('len dataset', len(self.dataset))
        print('test_ok', self.test_ok)
        print('test_false', self.test_false)

        self.log.write('len dataset: {}\n'.format(len(self.dataset)))
        self.log.write('test_ok: {}\n'.format(self.test_ok))
        self.log.write('test_false: {}\n'.format(self.test_false))


    def label_cut(self, c: Constr, type: CutType):

        ret = 1
        for sol in self.instance.solutions:
            total = 0
            for key, value in c.expr.items():
                total = total + sol[key.name]*value

            # print(total, c.sense, (-1) * c.const)
            # input()
            if c.sense == '<':
                if total - (-1) * c.const > self.instance.model.infeas_tol:
                    print('checking constr', c)
                    # for key, value in c.expr.items():
                    #     print(key, value, sol[key.name])
                    print('type {}: {} {} {} - {}\n'.format(type, total, c.sense, (-1) * c.const, total - (-1) * c.const))
                    self.log.write('type {}: {} {} {} - {}\n'.format(type, total, c.sense, (-1) * c.const, total - (-1) * c.const))
                    # input(0)
                    return 0
            elif c.sense == '>':
                if total - (-1) * c.const < (0 - self.instance.model.infeas_tol):
                    print('checking constr', c)
                    # for key, value in c.expr.items():
                    #     print(key, value, sol[key.name])
                    print('type {}: {} {} {} - {}\n'.format(type, total, c.sense, (-1) * c.const, total - (-1) * c.const))
                    self.log.write('type {}: {} {} {} - {}\n'.format(type, total, c.sense, (-1) * c.const, total - (-1) * c.const))
                    # input(0)
                    return 0

        return 1
