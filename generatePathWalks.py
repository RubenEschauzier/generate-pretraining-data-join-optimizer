import rdflib.term
from rdflib.graph import Graph
import random


def get_all_subject(g):
    all_subj = set()
    for (subj, _, _) in g:
        all_subj.add(subj)
    return all_subj


def get_start_triple_walk(start_point, g):
    all_triples = list(g.triples((start_point, None, None)))
    chosen_index = random.randrange(len(all_triples))
    return all_triples[chosen_index]


# Note: the walk v1 -> v2 -> v3 has v1 as tail and v3 as head
def extend_walk(g, head, tail, walk):
    triples_head = list(g.triples((head, None, None)))
    triples_tail = list(g.triples((None, None, tail)))

    all_triples = triples_head.copy()
    all_triples.extend(triples_tail)

    if len(all_triples) == 0:
        return head, tail, walk

    chosen_index = random.randrange(len(all_triples))
    # Determine whether we extend at head or tail
    if chosen_index >= len(triples_head):
        # If we get new tail, we insert at front of list to retain order of walk
        walk.insert(0, all_triples[chosen_index])
        new_tail = all_triples[chosen_index][0]
        return head, new_tail, walk
    # New head gets appended for similar reasons
    walk.append(all_triples[chosen_index])
    new_head = all_triples[chosen_index][2]
    return new_head, tail, walk


def generate_path_walks(g, start_points, repeats, max_size):
    all_walks = []
    for start in start_points:
        for i in range(repeats):
            start_triple = get_start_triple_walk(start, g)
            head = start_triple[2]
            tail = start_triple[0]
            size = random.randrange(2, max_size)
            walk = [start_triple]

            for j in range(size):
                head, tail, walk = extend_walk(g, head, tail, walk)

            if len(walk) > 1:
                all_walks.append(walk)
    return all_walks


def generate_path_queries(g, prob_non_variable):
    all_subj = get_all_subject(g)
    walks = generate_path_walks(g, all_subj, 10, 6)

    queries = []

    for walk in walks:
        variable_counter = 1
        term_to_variable_dict = {}
        print(walk)
        walk_query = "CONSTRUCT WHERE { "
        for i, triple in enumerate(walk):
            if triple[0] not in term_to_variable_dict:
                term_to_variable_dict[triple[0]] = "?v{}".format(variable_counter)
                variable_counter += 1
            if triple[2] not in term_to_variable_dict:
                term_to_variable_dict[triple[2]] = "?v{}".format(variable_counter)
                variable_counter += 1

            triple_string = "{} <{}> {} . ".format(
                term_to_variable_dict[triple[0]],
                triple[1],
                term_to_variable_dict[triple[2]]
            )
            if i == len(walk) - 1:
                r = random.uniform(0, 1)
                if r < prob_non_variable:
                    if type(triple[2]) == rdflib.term.Literal:
                        triple_string = "{} <{}> {} . ".format(
                            term_to_variable_dict[triple[0]],
                            triple[1],
                            triple[2]
                        )
                    # If not literal, we need brackets around it
                    else:
                        triple_string = "{} <{}> <{}> . ".format(
                            term_to_variable_dict[triple[0]],
                            triple[1],
                            triple[2]
                        )
            walk_query += triple_string
        walk_query += "}"
        queries.append([walk_query, len(walk)])
        print(walk)
        print(walk_query)
        break
    return queries


if __name__ == "__main__":
    input_graph = Graph()
    input_graph.parse("dataWatDiv/dataset.nt", format="nt")
    generated_path_queries = generate_path_queries(input_graph, .25)
    print(generated_path_queries)
    num_triples = 0
    for r in input_graph.query(generated_path_queries[0][0]):
        num_triples += 1
    print(num_triples)
    pass
