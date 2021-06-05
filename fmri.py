import sklearn.cluster
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from transformers import GPT2DoubleHeadsModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2 = GPT2DoubleHeadsModel.from_pretrained("distilgpt2")

def encode(sent):
	inp = tokenizer.encode(sent)
	outp = gpt2(torch.tensor([inp]), output_hidden_states=True)

	hidden_states = []
	for layer in outp.hidden_states:
		l = torch.tensor([x.detach().numpy() for x in layer])
		hidden_states.append(l)

	res = torch.tensor([x.detach().numpy() for x in hidden_states]).squeeze()

	return res.swapaxes(1, 0)

# sent = "I don't want to listen to Hidden Brain"
sents = [
	#"I don't want to listen to Hidden Brain",
	"Who are you?",
	"Where am I?",
	"Where do you like to spend your free time?",
	"I am Chris.",
	"You are here.",
	"And we can just not show the labels on the X axis."
	#"7. Me gustaría una cerveza."
]

wiki_sents = []
with open('all_wiki_test.txt', 'r') as f:
	wiki_sents = [x.strip() for x in f.readlines()]

def heatmap(sents):
		for sent in sents:
				sent_tok = [tokenizer.decode(x) for x in tokenizer.encode(sent)]
				(fig, axs) = plt.subplots(nrows=len(hidden_brain), ncols=1)
				axs = axs[::-1]

				hidden_avg = hidden_brain.mean(dim=0)

				global_min = hidden_brain.detach().numpy().min()
				global_max = hidden_brain.detach().numpy().max()

				for i in range(len(hidden_brain)):
					word = hidden_brain[i] - hidden_avg

					ax = axs[i]

					ax.set_ylabel(sent_tok[i])
					ax.set_xticks([])
					ax.set_xticklabels("")
					ax.set_yticks([])
					ax.set_yticklabels("")

					ax.imshow(word, cmap='jet', interpolation='nearest', origin='lower')

					ax.set_aspect(10)
				# fig.tight_layout()

		plt.show()

def rdm(sents):
	sent_vecs = np.array([encode(sent)[-1][-1].detach().numpy().squeeze() for sent in sents])
	print(sent_vecs.shape)

	'''
	sims = np.zeros((len(sent_vecs), len(sent_vecs)))
	for i in range(len(sent_vecs)):
		for j in range(len(sent_vecs)):
			sim = ((sent_vecs[i] - sent_vecs[j])**2).mean()
			sims[i][j] = sim
	'''

	sims = squareform(pdist(sent_vecs))

	(fig, ax) = plt.subplots()
	ax.set_xticks(range(len(sents)))
	ax.set_yticks(range(len(sents)))
	ax.set_xticklabels(sents)
	ax.set_yticklabels(sents)
	ax.xaxis.tick_top()
	plt.xticks(rotation=90)
	plt.imshow(sims, cmap='gray')
	plt.show()

def cluster(sents):
	sent_vecs = np.array([encode(sent)[-1][-1].detach().numpy().squeeze() for sent in sents])
	print(sent_vecs.shape)

	# clusterer = sklearn.cluster.KMeans(n_clusters=4)

	# res = clusterer.fit_transform(sent_vecs)

	scaled = StandardScaler().fit_transform(sent_vecs)
	reducer = umap.UMAP()
	embedding = reducer.fit_transform(scaled)
	

	(fig, ax) = plt.subplots()

	X = embedding[:, 0]
	Y = embedding[:, 1]

	ax.scatter(X, Y)

	for i in range(len(sents)):
		plt.annotate(sents[i], (X[i], Y[i]))

	plt.show()

	print(res)

rdm(wiki_sents)
