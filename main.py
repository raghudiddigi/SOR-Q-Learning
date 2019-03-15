
# coding: utf-8

# In[ ]:


import numpy as np
import mdptoolbox, mdptoolbox.example
import random
from numpy.random import seed



#np.random.seed(0)
#P, R = mdptoolbox.example.forest()
s = 10
a= 5
discount = 0.6
q_count = 0
sor_q_count = 0
policy_count = 0
sor_policy_count = 0
percentage_count = 0

episodes = 100
iterations = 100000

normal_total_diff = np.zeros((episodes,iterations))
sor_total_diff= np.zeros((episodes,iterations))


for count in range(episodes):
    
    np.random.seed((count+1)*100)
    random.seed((count+1)*110)
    
    P, R = mdptoolbox.example.rand(s, a)
    #print(P)
    #print(np.min(P))

    vi = mdptoolbox.mdp.ValueIteration(P, R, discount,epsilon=0.00001)
    vi.run()
    #print(vi.policy)

    #print('************************************')
    ql = mdptoolbox.mdp.QLearning(P, R, discount,n_iter=iterations)
    #ql.setVerbose()
    ql.run()
    #print(ql.V)
    #print(np.shape(vi.V) )
    #print(ql.policy,np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V))))
    #print('************************************')
    ql2 = mdptoolbox.mdp.SOR_QLearning(P, R, discount,n_iter=iterations)
    ql2.run()
    #print(ql2.V)
    q_count += np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V)))
    sor_q_count +=np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V)))
    
    policy_count += np.sum(vi.policy != ql.policy)
    sor_policy_count += np.sum(vi.policy != ql2.policy)
    
#     if np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V))) < np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V))):
#         percentage_count = percentage_count + 1
    #print(np.shape(vi.V),np.shape(ql.q_values))
    norm_diff = vi.V - ql.q_values
    normal_total_diff[count] = np.linalg.norm(norm_diff,axis =1)
    
    norm_diff_sor = vi.V - ql2.q_values
    sor_total_diff[count] = np.linalg.norm(norm_diff_sor,axis =1)
    
    print(np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V))),np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V))),np.sum(vi.policy != ql.policy),np.sum(vi.policy != ql2.policy),ql2.w)
    #print(np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V))),np.sum(vi.policy != ql2.policy),ql2.w)

