import sys
from instance import Instance
from extract_features import ExtractFeatures
import random
from mip.constants import CutType
from func_timeout import func_timeout, FunctionTimedOut

file = sys.argv[1]
print('instance name: ', file)
instance = Instance(file)
extractor = ExtractFeatures(instance)

combinatorial_cuts = [CutType.CLIQUE, CutType.KNAPSACK_COVER, CutType.ODD_WHEEL, CutType.PROBING]
noncombinatorial_cuts = [CutType.GOMORY, CutType.LIFT_AND_PROJECT, CutType.MIR, CutType.FLOW_COVER,
                         CutType.GMI, CutType.LATWO_MIR, CutType.RED_SPLIT, CutType.ZERO_HALF,
                         CutType.RED_SPLIT_G, CutType.RESIDUAL_CAPACITY, CutType.TWO_MIR]

iteration = 0
erros = 0

while iteration < 5 and erros < 6000:
    iteration = iteration + 1
    random.shuffle(combinatorial_cuts)
    random.shuffle(noncombinatorial_cuts)
    size_comb_cuts = random.randint(0, len(combinatorial_cuts))
    size_noncomb_cuts = random.randint(1, len(noncombinatorial_cuts))
    print(combinatorial_cuts, noncombinatorial_cuts)
    print(size_comb_cuts, size_noncomb_cuts)
    try:
        if size_comb_cuts > 0:
            doitReturnValue = func_timeout(14400, extractor.test, [combinatorial_cuts[0:size_comb_cuts], noncombinatorial_cuts[0:size_noncomb_cuts]])
            erros = erros + doitReturnValue
        else:
            doitReturnValue = func_timeout(14400, extractor.test, [[], noncombinatorial_cuts[0:size_noncomb_cuts]])
            erros = erros + doitReturnValue
    except FunctionTimedOut:
        extractor.writeEnd()
        print('timeout')
    print('-----------------------------------------')