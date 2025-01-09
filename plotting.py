
import matplotlib.pylab as plt
from sklearn.manifold import TSNE

import iars
#model name is the model used, ts2vec original or proposed
def tsne_plot(net,x,label,model_name,dir_used,count,data_set):
    o = net(x)
    o = o.reshape(x.shape[0],-1)
    #o = o.mean(dim=1)
    print(o.size())

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    o = o.squeeze()
    X_train_tsne = tsne.fit_transform(o.cpu().detach().numpy())

    # Data distribution Plotting
    plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=label, alpha=0.8,cmap='viridis')
    plt.title('t-SNE visualization of training data')
    plt.colorbar()

    plt.savefig(f"{dir_used}/tsne_{data_set}_without_pool_{model_name}_{count}.png")

    plt.show()


def tsne_plot_avg(net,x,label,model_name,dir_used,count,data_set):
    o = net(x)
    print(o.shape)
    #o = o.reshape(x.shape[0],-1)
    o = o.mean(dim=1)
    print(o.size())

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    o = o.squeeze()
    X_train_tsne = tsne.fit_transform(o.cpu().detach().numpy())

    # Data distribution Plotting
    plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=label, alpha=0.8,cmap='viridis')
    plt.title('t-SNE visualization of training data')
    plt.colorbar()

    plt.savefig(f"{dir_used}/tsne_{data_set}_avgpool_{model_name}_{count}.png")

    plt.show()
    return o

