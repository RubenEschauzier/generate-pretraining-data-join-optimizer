import numpy as np
import random
import rdflib.term
from scipy import spatial
from rdflib.graph import Graph
g = Graph()
g.parse("dataBerlin/dataset.nt", format="nt")
predicates_vectors = []
predicates_vectors_dict = {}
for (subj, pred, obj) in g:
    predicates_vectors.append(str(pred))
    predicates_vectors_dict[str(pred)] = 0

vectors = []
predicates = []
with open('dataBerlin/vectorsSmall/vectors.txt') as f:
    for line in f.readlines():
        predicate = line.split('[sep]')[0]
        vector = np.array(line.split('[sep]')[1].strip().split(' '), dtype=float)

        vectors.append(vector)
        predicates.append(predicate)

break_it = False
to_evaluate_predicates_high = []
to_evaluate_predicates_low = []
for i, vec in enumerate(vectors):
    if np.sum(vec) != 0 and predicates[i] in predicates_vectors_dict:
        for j in range(len(vectors)):
            if i != j and predicates[j] in predicates_vectors_dict:
                if spatial.distance.cosine(vec, vectors[j]) > .8:
                    to_evaluate_predicates_high.append((predicates[i], predicates[j]))
                if spatial.distance.cosine(vec, vectors[j]) < .2:
                    to_evaluate_predicates_low.append((predicates[i], predicates[j]))
# low_predicates = random.sample(to_evaluate_predicates_low, 50)
# high_predicates = random.sample(to_evaluate_predicates_high, 50)
low_predicates=to_evaluate_predicates_low
high_predicates=to_evaluate_predicates_high

# queries_low = []
# queries_high = []
# for low_predicate in low_predicates:
#     stringSingle1 = "SELECT ?x ?y WHERE {?x <" + str(low_predicate[0]) + "> ?y}"
#     stringSingle2 = "SELECT ?x ?y WHERE {?x <" + str(low_predicate[1]) + "> ?y}"
#     stringJoin = "SELECT ?x ?y ?z WHERE {?x <" + str(low_predicate[0]) + "> ?y . ?x <" + str(
#         low_predicate[1]) + "> ?z}"
#     queries_low.append([stringSingle1, stringSingle2, stringJoin])
#
# for high_predicate in high_predicates:
#     stringSingle1 = "SELECT ?x ?y WHERE {?x <" + str(high_predicate[0]) + "> ?y}"
#     stringSingle2 = "SELECT ?x ?y WHERE {?x <" + str(high_predicate[1]) + "> ?y}"
#     stringJoin = "SELECT ?x ?y ?z WHERE {?x <" + str(high_predicate[0]) + "> ?y . ?x <" + str(
#         high_predicate[1]) + "> ?z}"
#     queries_high.append([stringSingle1, stringSingle2, stringJoin])
#
# for i, query_low in enumerate(queries_low):
#     with open("testQueriesRDF2Vec/LowQueries/query{}".format(i), 'a') as f:
#         for query in query_low:
#             f.write(query)
#             f.write('[sep]')
# print("Done!")
# for i, query_high in enumerate(queries_high):
#     with open("testQueriesRDF2Vec/HighQueries/query{}".format(i), 'a') as f:
#         for query in query_high:
#             print(query)
#             f.write(query)
#             f.write('[sep]')
# print(queries_low)
queries_low = []
queries_high = []
for low_predicate in low_predicates:
    stringSingle1 = "SELECT ?x ?y WHERE {?x <" + str(low_predicate[0]) + "> ?y}"
    stringSingle2 = "SELECT ?x ?y WHERE {?x <" + str(low_predicate[1]) + "> ?y}"
    stringJoin = "SELECT ?x ?y ?z WHERE {?x <" + str(low_predicate[0]) + "> ?y . ?y <" + str(
        low_predicate[1]) + "> ?z}"
    queries_low.append([stringSingle1, stringSingle2, stringJoin])

for high_predicate in high_predicates:
    stringSingle1 = "SELECT ?x ?y WHERE {?x <" + str(high_predicate[0]) + "> ?y}"
    stringSingle2 = "SELECT ?x ?y WHERE {?x <" + str(high_predicate[1]) + "> ?y}"
    stringJoin = "SELECT ?x ?y ?z WHERE {?x <" + str(high_predicate[0]) + "> ?y . ?y <" + str(
        high_predicate[1]) + "> ?z}"
    queries_high.append([stringSingle1, stringSingle2, stringJoin])

for i, query_low in enumerate(queries_low):
    with open("testQueriesChainRDF2VecBerlin/LowQueries/query{}".format(i), 'a') as f:
        for query in query_low:
            f.write(query)
            f.write('[sep]')
for i, query_high in enumerate(queries_high):
    with open("testQueriesChainRDF2VecBerlin/HighQueries/query{}".format(i), 'a') as f:
        for query in query_high:
            f.write(query)
            f.write('[sep]')

# rdf_predicates = [rdflib.term.URIRef(x) for x in to_evaluate_predicates]
# triples_with_predicate_1 = list(g.triples((None, rdf_predicates[0], None)))
# triples_with_predicate_2 = list(g.triples((None, rdf_predicates[1], None)))
# triples_to_test1 = random.sample(triples_with_predicate_1, 5)
# triples_to_test2 = random.sample(triples_with_predicate_2, 5)
# print(triples_to_test1)


