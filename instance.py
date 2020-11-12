from mip import Model
import gzip
import shutil
import os
from datetime import datetime
from mip.entities import Var


class Instance:
    def __init__(self, filename: str):
        self.name = filename.split('/')[1].split('.')[0]
        with gzip.open(filename, 'rb') as f_in:
            with open('{}.mps'.format(self.name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        self.model = Model(solver_name='cbc')
        self.model.verbose = 0
        self.model.seed = datetime.now()  # random seed

        self.model.read('{}.mps'.format(self.name))
        if os.path.isfile('{}/{}.bas'.format(filename.split('/')[0], self.name)):
            self.model.read('{}/{}.bas'.format(filename.split('/')[0], self.name))
        #     input('bas read')
        self.model.verbose = 0
        self.model.write('test.lp')
        input()
        self.solutions_file = []
        self.solutions = []
        for sub in os.listdir('solutions/{}'.format(self.name)):
            f = 'solutions/{}/{}/{}.sol.gz'.format(self.name, sub, self.name)
            with gzip.open(f, 'rb') as f_in:
                name = '{}_{}.sol'.format(self.name, sub)
                white = 0
                size = 0
                for line in f_in:
                    aux = line.strip().decode("utf8")
                    aux = aux.rstrip().lstrip()
                    if aux.strip(' \n') != '':
                        white = white + 1
                f_in.seek(0)
                if white > 0:
                    sol = {x.name: 0 for x in self.model.vars}
                    with open(name, 'w') as f_out:
                        i = 0
                        for line in f_in:
                            utf8_in = line.strip().decode("utf8")
                            # input(utf8_in.find('=obj='))
                            if utf8_in.find('=obj=') > 0:
                                f_out.write('{}\n'.format(utf8_in))
                                continue

                            aux = line.strip().decode("utf8")
                            aux = aux.rstrip().lstrip()
                            aux = " ".join(aux.split())
                            lc = aux.split(" ")
                            if len(lc) < 2 or lc[0] == '#':
                                continue
                            if lc[0] == '=obj=':
                                f_out.write('{}\n'.format(utf8_in))
                                continue
                            if lc[0] not in sol:
                                msg = 'Instância {} - Variável {} não existente no modelo'.format(self.name, lc[0])
                                raise NameError(msg)
                            f_out.write('{}\t{}\t{}\n'.format(i, lc[0], float(lc[1])))
                            sol[lc[0]] = float(lc[1])
                            i = i + 1
                        self.solutions.append(sol)
        print(len(self.solutions))
        if len(self.solutions) < 1:
            raise NameError('Instância {} - Não há soluções válidas'.format(self.name))
            # input(self.solutions)
    # def __del__(self):
    #     print('destructor')
    #     os.remove('{}.mps'.format(self.name))
    #     for sol in self.solutions_file:
    #         os.remove(sol)
