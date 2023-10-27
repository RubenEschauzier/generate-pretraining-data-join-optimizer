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
        print("No triple")
        return head, tail, walk

    chosen_index = random.randrange(len(all_triples))
    # Determine whether we extend at head or tail
    walk.append(all_triples[chosen_index])
    if chosen_index >= len(triples_head):
        new_tail = all_triples[chosen_index][0]
        return head, new_tail, walk
    new_head = all_triples[chosen_index][2]
    return new_head, tail, walk


def generate_path_walks(g, start_points, repeats, max_size):
    for start in start_points:
        for i in range(repeats):
            start_triple = get_start_triple_walk(start, g)
            size = random.randrange(2, max_size)
            walk = [start_triple]
            for j in range(size):
                head, tail, walk = extend_walk(g, start_triple[2], start_triple[0], walk)
                print(len(walk))
                print(walk)
                pass
            if len(walk) > 1:
                pass

        pass


def generate_path_queries(g):
    all_subj = get_all_subject(g)
    walks = generate_path_walks(g, all_subj, 10, 6)
    pass


if __name__ == "__main__":
    input_graph = Graph()
    input_graph.parse("dataWatDiv/dataset.nt", format="nt")
    generate_path_queries(input_graph)
    pass
