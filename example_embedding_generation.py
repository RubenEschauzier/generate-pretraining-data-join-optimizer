from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# corpus = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'p1', 'p2', 'p3', 'p4']
# walks = [['x1', 'p1', 'x2', 'p2', 'x3'], ['x4', 'p3', 'x5', 'p4', 'x6']]
#
# vector_dim = 2
# model = Word2Vec(walks, min_count=1, window=6, vector_size=vector_dim, epochs=200)
# x = []
# y = []
# for token in corpus:
#     vector = model.wv[token]
#     x.append(vector[0])
#     y.append(vector[1])
#
#
# fig, ax = plt.subplots()
# ax.scatter(x, y)
#
# for i, txt in enumerate(corpus):
#     ax.annotate(txt, (x[i], y[i]))
#
# plt.show()

x = [0.05, 0.05, 0.2, 0.15, 0.1,
     0.25, 0.28,  0.3, 0.35, 0.35]
y = [0.78, 0.9, 0.8, 0.85, 0.8,
     0.37, 0.2, 0.3, 0.25, 0.35]
tags = ['  x1', '  p1', '  x3', '  p2', '  x2',
        '  x4', '  p3', '  x5', ' p4', ' x6']
with plt.style.context('classic'):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=80)

    for i, txt in enumerate(tags):
        ax.annotate(txt, (x[i], y[i]), fontsize=20)
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("example_embedding.pdf", transparent=True)
    plt.show()


