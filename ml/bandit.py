import  numpy as np
import  pymc
#wins 和 trials 是一个N维向量，N是赌博机的臂的个数，每个元素记录了
choice = np.argmax(pymc.rbeta(1 + wins, 1 + trials - wins))
wins[choice] += 1
trials += 1