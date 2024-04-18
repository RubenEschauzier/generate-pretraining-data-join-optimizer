import random
import rdflib.term
from src.random_query_generation.utils import filter_isomorphic_queries, update_counts_predicates_used, \
    choose_triple_weighted, get_start_triple_walk, get_all_subject, track_equivalent_predicates, \
    filter_equivalent_queries


# Note: the walk v1 -> v2 -> v3 has v1 as tail and v3 as head
def extend_walk_path(g, head, tail, walk, predicate_usage_counts=None):
    triples_head = list(g.triples((head, None, None)))
    triples_tail = list(g.triples((None, None, tail)))

    all_triples = triples_head.copy()
    all_triples.extend(triples_tail)

    if len(all_triples) == 0:
        return head, tail, walk, predicate_usage_counts

    # If we pass a predicate_usage_count dictionary we should do weighted sampling
    if predicate_usage_counts is not None:
        chosen_index, predicate_usage_counts = choose_triple_weighted(
            all_triples, predicate_usage_counts)
        predicate_usage_counts = update_counts_predicates_used(
            all_triples, chosen_index, predicate_usage_counts)
    else:
        # Choose with equal weight
        chosen_index = random.randrange(len(all_triples))
    # chosen_index = random.randrange(len(all_triples))
    # Determine whether we extend at head or tail
    if chosen_index >= len(triples_head):
        # If we get new tail, we insert at front of list to retain order of walk
        walk.insert(0, all_triples[chosen_index])
        new_tail = all_triples[chosen_index][0]
        return head, new_tail, walk, predicate_usage_counts
    # New head gets appended for similar reasons
    walk.append(all_triples[chosen_index])
    new_head = all_triples[chosen_index][2]
    return new_head, tail, walk, predicate_usage_counts


def generate_path_walks(g, start_points, repeats, max_size):
    predicate_usage_counts = {}
    all_walks = []
    for start in start_points:
        for i in range(repeats):
            start_triple = get_start_triple_walk(start, g)
            head = start_triple[2]
            tail = start_triple[0]
            size = random.randrange(2, max_size)
            walk = [start_triple]

            for j in range(size):
                head, tail, walk, predicate_usage_counts = extend_walk_path(
                    g, head, tail, walk, predicate_usage_counts)

            if len(walk) > 1:
                all_walks.append(walk)
    return all_walks


def generate_path_queries(g, repeats, max_walk_size, prob_non_variable):
    all_subj = get_all_subject(g)
    walks = generate_path_walks(g, all_subj, repeats, max_walk_size)

    queries = []
    used_predicates = set()
    used_predicates_dict = {}

    for walk in walks:
        predicates = tuple(sorted([triple[1] for triple in walk]))
        equivalent_predicates = track_equivalent_predicates(used_predicates, predicates)
        variable_counter = 1
        non_variable_edges = 0
        term_to_variable_dict = {}
        walk_query = "SELECT * WHERE { "
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
                    non_variable_edges += 1

                    # If literal at end we don't add brackets, because not URI
                    if type(triple[2]) == rdflib.term.Literal:
                        triple_string = "{} <{}> \"{}\" . ".format(
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
        # queries.append(walk_query)
        queries, used_predicates_dict = \
            filter_equivalent_queries(walk_query,
                                      walk,
                                      queries,
                                      equivalent_predicates,
                                      used_predicates_dict,
                                      predicates,
                                      non_variable_edges)
    # filter_isomorphic_queries([query[0] for query in queries])
    print("Filter path queries")
    queries = filter_isomorphic_queries(queries)
    return queries


