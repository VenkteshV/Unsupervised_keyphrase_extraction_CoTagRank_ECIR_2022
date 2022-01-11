import pandas as pd
import os 
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
def plot_results(data):
    plt.figure(figsize=(40  ,3))

    fig, ax = plt.subplots()
    ax.set_ylabel('F1')
    # ax.set_xlabel('LDA embeddings dimension')
    # ax.set_xlabel('Window size, w')
    ax.set_xlabel('damping factor, λ')
    ax.plot(data['damping'],data['Inspec_CoTagRankWindow'],'r-s', label = 'CoTagRankWindow')
    ax.plot(data['damping'], data['Inspec_CoTagRankUSE'],'g-o', label = 'CoTagRank')
    ax.plot(data['damping'],data['Inspec_CoTagRanks2v'],'b-d', label='CoTagRanks2v')
    ax.legend(loc='best')
    plt.savefig('result_damping.png')
    plt.show()




if __name__=='__main__':
    data = pd.read_csv(dir_path+"/result_damping_Inspec.csv")
    plot_results(data)