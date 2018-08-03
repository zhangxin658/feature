import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA (real number)//变量个数
DNA_BOUND = [0, 5]       # solution upper and lower bounds//参数选取范围
N_GENERATIONS = 200      # 迭代数量
POP_SIZE = 100           # population size//种族数量
N_KID = 50               # n kids per generation //每一代有n个孩子


def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function//


# find non-zero fitness for selection
def get_fitness(pred): return pred.flatten()    #返回适应度值


def make_kid(pop, n_kid):  #传入的参数包括父代节点和要生成孩子的个数
    # generate empty kid holder //
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}#生成孩子参数
    kids['mut_strength'] = np.empty_like(kids['DNA']) #生成孩子的变异强敌
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):  #在
        # crossover (roughly half p1 and half p2)
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)#产生随机采样,从种群中找到两个，并且不能重复
        print('p1, p2:', p1, p2)
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        print('cp:', cp)
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        kv += ks * np.random.randn(*kv.shape)
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
    return kids


def kill_bad(pop, kids):
    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = get_fitness(F(pop['DNA']))            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop


pop = dict(DNA=np.random.rand(DNA_SIZE, DNA_SIZE).repeat(POP_SIZE, axis=0),   # initialize the pop DNA values //初始化参数；
           mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values//初始化变异强度；
print(pop)
# plt.ion()       # something about plotting
# x = np.linspace(*DNA_BOUND, 200)
# plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    # # something about plotting
    # if 'sca' in globals(): sca.remove()
    # sca = plt.scatter(pop['DNA'], F(pop['DNA']), s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # ES part
    kids = make_kid(pop, N_KID)  # 通过父代生成子代
    pop = kill_bad(pop, kids)   # keep some good parent for elitism

# plt.ioff()
# plt.show()