
from rdflib.graph import Graph, URIRef
from pyrdf2vec import RDF2VecTransformer
from gensim.models import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker
import numpy as np
import itertools
import random
from tqdm import tqdm

# Define parameters run
num_sim_pred = 16384
# Num triples to visit during walk
depth_walk = 2
kg = KG("dataWatDiv/dataset.nt", fmt="nt")

g = Graph()
g.parse("dataWatDiv/dataset.nt", format="nt")

# Get all predicates and entities
all_predicate_occurence = []
predicates = set()
entities = set()
predicatesURI = set()

for (subj, pred, obj) in g:
    predicates.add(str(pred))
    predicatesURI.add(pred)
    entities.add(subj)
    entities.add(obj)
    all_predicate_occurence.append(pred)

predicates = list(predicates)
predicatesURI = list(predicatesURI)
all_walks = []
num_pred_done = 0
for predicate in predicatesURI:
    print(num_pred_done)
    num_pred_done += 1
    triples_with_predicate = g.triples((None, predicate, None))
    tp = list(triples_with_predicate)
    num_walks_predicate = 0
    for i in range(num_sim_pred):
        num_triples_chosen = 0
        walk_in_progress = []
        # Choose random triple from our list of triples
        index = random.randrange(len(tp))
        startTriple = tp[index]
        startSubject = startTriple[0]
        # This is the triple that the subject belongs to, for ordering
        startSubjectIndex = 0
        startObject = startTriple[2]
        # This is the triple that the object belongs to, for ordering
        startObjectIndex = 0
        walk_in_progress.extend(list(startTriple))

        while num_triples_chosen < depth_walk:
            # Get all triples with starting object as subject
            start_object_as_subject = list(g.triples((startObject, None, None)))
            # All triples with starting subject as object
            start_subject_as_object = list(g.triples((None, None, startSubject)))
            # Merge all possible triples
            possibleTriples = list(itertools.chain(start_object_as_subject, start_subject_as_object))
            if len(possibleTriples) == 0:
                break
            # Get new triple to walk to
            newIndex = random.randrange(len(possibleTriples))
            startTriple = possibleTriples[newIndex]
            if newIndex < len(start_object_as_subject):
                startObject = startTriple[2]
                walk_in_progress.extend(list(startTriple))
            else:
                extendedWalk = list(startTriple)
                startSubject = startTriple[0]
                extendedWalk.extend(walk_in_progress)
                walk_in_progress = extendedWalk

            # walk_in_progress.extend(list(startTriple))
            num_triples_chosen += 1

        if len(walk_in_progress) == depth_walk * 3 + 3:
            # Ugliest way to remove duplicate entries from walking, should do it in walk generation
            lastElement = walk_in_progress[-1]
            del walk_in_progress[2::3]
            walk_in_progress.append(lastElement)
            all_walks.append(walk_in_progress)
            num_walks_predicate += 1

entities = list(entities)
corpus = [[str(word) for word in walk] for walk in all_walks]
vector_dim = 128
model = Word2Vec(corpus, min_count=1, window=5, vector_size=vector_dim, epochs=100)
with open('dataWatDiv/vectorsWatDiv/vectors_depth_2.txt', 'w') as f:
    # First write predicates to file
    for predicate in predicates:
        if predicate in model.wv:
            to_write = str(predicate) + '[sep]' + ' '.join([str(x) for x in model.wv[predicate]])
            f.write(to_write)
            f.write('\n')
            pass
        else:
            to_write = str(predicate) + '[sep]' + ' '.join(['0' for i in range(vector_dim)])
            f.write(to_write)
            f.write('\n')
    # Write non-predicates to file
    for key in model.wv.index_to_key:
        if key not in predicates:
            to_write = str(key) + '[sep]' + ' '.join([str(x) for x in model.wv[key]])
            f.write(to_write)
            f.write('\n')

# unique, counts = np.unique(x, return_counts=True)

# for entity in entities:
#     if Vertex(str(entity)) not in kg._entities:
#         pass
#     else:
#         print("IN THERE")
#
# transformer = RDF2VecTransformer(
#     Word2Vec(epochs=10),
#     walkers=[RandomWalker(4, 10, with_reverse=False, n_jobs=2)],
#     # verbose=1
# )
# embeddings, literals = transformer.fit_transform(kg, entities)
