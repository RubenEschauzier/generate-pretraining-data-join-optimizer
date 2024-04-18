import math
from sklearn.preprocessing import minmax_scale
import rdflib.term
from rdflib.graph import Graph
from rdflib.compare import to_isomorphic, IsomorphicGraph
import random
import re
from tqdm import tqdm
import itertools
import numpy as np
import networkx as nx


def get_all_subject(g):
    all_subj = set()
    for (subj, _, _) in g:
        all_subj.add(subj)
    return all_subj


def get_all_object(g):
    all_obj = set()
    for (_, _, obj) in g:
        all_obj.add(obj)
    return all_obj


def get_start_triple_walk(start_point, g):
    all_triples = list(g.triples((start_point, None, None)))
    chosen_index = random.randrange(len(all_triples))
    return all_triples[chosen_index]


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


def weighted_choice(weights):
    return random.choices(range(len(weights)), weights=weights)[0]


def choose_triple_weighted(all_triples, predicate_usage_counts):
    weights = []
    for triple in all_triples:
        predicate = triple[1]
        if predicate in predicate_usage_counts:
            weights.append(-(predicate_usage_counts[predicate] + 1))
        else:
            weights.append(-1)
    weights = np.array(weights)
    # Min-max normalization for numerical stability for very negative weights
    weights = minmax_scale(weights)

    # Choose according to weights
    # TODO: CHECK IF NUMERICAL STABILITY WORKS AND THAT THE DESIRED EFFECT (LESS PROB FOR HIGH FREQ)
    # TODO: IS MAINTAINED
    probabilities = np.exp(weights) / sum(np.exp(weights))
    chosen_index = weighted_choice(probabilities)
    return chosen_index, predicate_usage_counts


def update_counts_predicates_used(all_triples, chosen_index, predicate_usage_counts):
    chosen_predicate = all_triples[chosen_index][1]

    # Update the counts
    if chosen_predicate in predicate_usage_counts:
        predicate_usage_counts[chosen_predicate] += 1
    else:
        predicate_usage_counts[chosen_predicate] = 1
    return predicate_usage_counts


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


class WrapperIsomorphicGraphFixedHashing:
    def __init__(self, graph):
        self.iso_graph = graph

    def __hash__(self):
        return self.iso_graph.internal_hash()

    def __eq__(self, other):
        return True


def filter_isomorphic_queries(queries):
    accepted_iso_set = set()
    accepted_queries = []

    for query in queries:
        iso = to_isomorphic(convert_query_to_graph(query))
        if WrapperIsomorphicGraphFixedHashing(iso) in accepted_iso_set:
            continue
        accepted_iso_set.add(WrapperIsomorphicGraphFixedHashing(iso))
        accepted_queries.append(query)
    print("FILTERING")
    print(len(queries))
    print(len(accepted_queries))
    return accepted_queries


def convert_query_to_graph(query):
    g = Graph()
    bgp = re.search('\{(.*?)\}', query).group().replace("}", "").replace("{", "")
    tps = bgp.split(" . ")[:-1]
    for tp in tps:
        spo = tp.strip().split(' ')
        triple_to_add = [rdflib.term.BNode(spo[0]), rdflib.term.URIRef(spo[1][1:-1])]
        if spo[2][0] == "?":
            triple_to_add.append(rdflib.term.BNode(spo[2]))
        else:
            triple_to_add.append(rdflib.term.Literal(spo[2]))
        g.add((triple_to_add[0], triple_to_add[1], triple_to_add[2]))
    return g


def generate_star_queries(g, repeats, max_size_star, prob_non_variable_edge):
    all_subj = get_all_subject(g)
    all_obj = get_all_object(g)

    # Get all different star walks
    predicate_usage_counts_subject_centre = {}
    predicate_usage_counts_object_centre = {}

    star_walks_subject_centre = generate_star_walks_subject_centre(g, all_subj, repeats, max_size_star,
                                                                   predicate_usage_counts_subject_centre)
    star_walks_object_centre = generate_star_walks_object_centre(g, all_obj, int(math.floor(repeats / 2)),
                                                                 max_size_star,
                                                                 predicate_usage_counts_object_centre)

    queries_subject_centre = []
    queries_object_centre = []
    used_predicates_subject_centre = set()
    used_predicates_subject_centre_dict = {}
    used_predicates_object_centre = set()
    used_predicates_object_centre_dict = {}
    # Generate subject_centre queries
    for walk_subject_centre in star_walks_subject_centre:
        predicates = tuple(sorted([triple[1] for triple in walk_subject_centre]))
        equivalent_predicates = track_equivalent_predicates(used_predicates_subject_centre, predicates)
        variable_counter = 1
        non_variable_edges = 0
        term_to_variable_dict = {}
        walk_query = "SELECT * WHERE { "
        for i, triple in enumerate(walk_subject_centre):
            # Handle logic for creation variables in the dictionary
            if triple[0] not in term_to_variable_dict:
                term_to_variable_dict[triple[0]] = "?v{}".format(variable_counter)
                variable_counter += 1

            r = random.uniform(0, 1)
            if r > prob_non_variable_edge:
                if triple[2] not in term_to_variable_dict:
                    term_to_variable_dict[triple[2]] = "?v{}".format(variable_counter)
                    variable_counter += 1
            triple_string, non_variable_edge = generate_triple_string(triple, r, prob_non_variable_edge,
                                                                      term_to_variable_dict,
                                                                      object_centre=False)
            if non_variable_edge:
                non_variable_edges += 1
            walk_query += triple_string
        walk_query += "}"

        # Filter out queries that have identical predicates and no named nodes / literals or
        # do have named nodes / literals but have been added many times with probability
        # that decrease exponentially by the number of times the predicates have been added to the
        # queries
        queries_subject_centre, used_predicates_subject_centre_dict = \
            filter_equivalent_queries(walk_query,
                                      walk_subject_centre,
                                      queries_subject_centre,
                                      equivalent_predicates,
                                      used_predicates_subject_centre_dict,
                                      predicates,
                                      non_variable_edges)
    print("Filter subject centre star queries")
    queries_subject_centre = filter_isomorphic_queries(queries_subject_centre)

    for walk_obj_centre in star_walks_object_centre:
        predicates = tuple(sorted([triple[1] for triple in walk_obj_centre]))
        equivalent_predicates = track_equivalent_predicates(used_predicates_object_centre, predicates)
        variable_counter = 1
        non_variable_edges = 0
        term_to_variable_dict = {}
        walk_query = "SELECT * WHERE { "
        for i, triple in enumerate(walk_obj_centre):
            # Handle logic for creation variables in the dictionary, object centres are only variables
            if triple[2] not in term_to_variable_dict:
                term_to_variable_dict[triple[2]] = "?v{}".format(variable_counter)
                variable_counter += 1
            if triple[0] not in term_to_variable_dict:
                term_to_variable_dict[triple[0]] = "?v{}".format(variable_counter)
                variable_counter += 1
            # Here we set r to arbitrary one, indicating that it will never have literals in the query
            triple_string, non_variable_edge = generate_triple_string(triple, 1, prob_non_variable_edge,
                                                                      term_to_variable_dict,
                                                                      object_centre=True)
            if non_variable_edge:
                non_variable_edges += 1

            walk_query += triple_string
        walk_query += "}"

        queries_object_centre, used_predicates_object_centre_dict = \
            filter_equivalent_queries(walk_query,
                                      walk_obj_centre,
                                      queries_object_centre,
                                      equivalent_predicates,
                                      used_predicates_object_centre_dict,
                                      predicates,
                                      non_variable_edges)

    print("Filtering object centre star queries")
    queries_object_centre = filter_isomorphic_queries(queries_object_centre)

    return queries_subject_centre, queries_object_centre


# Separate the logic for variable dict and string creation
def generate_triple_string(triple, r, prob_non_variable_edge, term_to_variable_dict, object_centre):
    non_variable_edge = False
    if r > prob_non_variable_edge or object_centre:
        triple_string = "{} <{}> {} . ".format(
            term_to_variable_dict[triple[0]],
            triple[1],
            term_to_variable_dict[triple[2]]
        )
    elif object_centre:
        non_variable_edge = True
        if type(triple[0]) == rdflib.term.Literal:
            triple_string = "\"{}\" <{}> {} . ".format(
                triple[1],
                triple[2],
                term_to_variable_dict[triple[0]]
            )
        # If not literal, we need brackets around it
        else:
            triple_string = "<{}> <{}> {} . ".format(
                triple[1],
                triple[2],
                term_to_variable_dict[triple[0]]
            )
    else:
        non_variable_edge = True
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
    return triple_string, non_variable_edge


def generate_star_walks_subject_centre(g, start_points, repeats, max_points,
                                       predicate_usage_counts, triples_to_ignore=None):
    all_walks = []
    for star_centre in start_points:
        for i in range(repeats):
            possible_star_points = get_possible_star_points_subject_centre(g, star_centre, triples_to_ignore)
            all_walks = extend_star_walk(possible_star_points, max_points, all_walks,
                                         predicate_usage_counts)
    return all_walks


def generate_star_walks_object_centre(g, start_points, repeats, max_points,
                                      predicate_usage_counts, triples_to_ignore=None):
    all_walks = []
    for star_centre in start_points:
        for i in range(repeats):
            possible_star_points = get_possible_star_points_object_centre(g, star_centre, triples_to_ignore)
            all_walks = extend_star_walk(possible_star_points, max_points, all_walks,
                                         predicate_usage_counts)
    return all_walks


def extend_star_walk(possible_star_points, max_points, all_walks, predicate_usage_counts):
    walk = []
    size = random.randrange(2, max_points)

    for j in range(size):
        if len(possible_star_points) == 0:
            break
        new_star_point, possible_star_points, predicate_usage_counts = \
            sample_new_star_point(possible_star_points, predicate_usage_counts)
        walk.append(new_star_point)

    if len(walk) > 1:
        all_walks.append(walk)

    return all_walks


# TODO: Docstring, note triple_to_ignore denotes the triple that connects multiple star queries together
def get_possible_star_points_subject_centre(g, star_centre, points_to_ignore=None):
    all_triples = list(g.triples((star_centre, None, None)))
    if points_to_ignore:
        for point in points_to_ignore:
            if point in all_triples:
                all_triples.remove(point)
    return all_triples


def get_possible_star_points_object_centre(g, star_centre, points_to_ignore=None):
    all_triples = list(g.triples((None, None, star_centre)))
    if points_to_ignore:
        for point in points_to_ignore:
            if point in all_triples:
                all_triples.remove(point)
    return all_triples


def sample_new_star_point(all_triples, predicate_usage_counts=None):
    if predicate_usage_counts is not None:
        chosen_index, predicate_usage_counts = choose_triple_weighted(
            all_triples, predicate_usage_counts)
        predicate_usage_counts = update_counts_predicates_used(
            all_triples, chosen_index, predicate_usage_counts)
    else:
        chosen_index = random.randrange(len(all_triples))
    returned_triple = all_triples[chosen_index]
    all_triples.remove(returned_triple)
    return returned_triple, all_triples, predicate_usage_counts


# The idea is that if the set of predicates is equivalent to something already used and it is the same shape, the query
# is superfluous. Here the order does not matter
def track_equivalent_predicates(used_predicate_set, predicates):
    if predicates in used_predicate_set:
        return True
    used_predicate_set.add(tuple(predicates))
    return False


def filter_equivalent_queries(walk_query,
                              walk,
                              all_queries,
                              equivalent_predicates,
                              used_predicates_dict,
                              used_predicates,
                              non_variable_edges, ):
    if not equivalent_predicates:
        used_predicates_dict[used_predicates] = 1
        all_queries.append(walk_query)
    elif non_variable_edges > 0:
        # We scale the acceptance chance of a predicate combination that we've seen before with one literal with a
        # quasi-exponential function to prevent our queries primarily consisting of often occurring paths in the
        # graph
        n_usages = used_predicates_dict[used_predicates]
        accept_prob = 1 / math.exp(n_usages)
        r = random.uniform(0, 1)
        if r < accept_prob:
            all_queries.append(walk_query)
            used_predicates_dict[used_predicates] += 1
    return all_queries, used_predicates_dict


# def generate_star_path_walks(g, start_points, repeats, max_stars_in_walk, max_size_star, max_size_path, p_object_star):
#     # 4 options for starting a walk from a star. You either make object star with subject or subject star with subject.
#     # Then, if that doesn't work you try to make an object star with object or subject star with object
#     # For continuing the walk, we can't start from our start_point, so we should choose either a random subject
#     # from an object star, or choose a random object from a subject star. In all case we should check whether we
#     # actually created a valid complex query with at least a path and star
#     all_walks = []
#     for start_point in start_points:
#
#         for i in range(repeats):
#             start_triple = get_start_triple_walk(start_point, g)
#             # For simplicity, we simply generate all star walks we can and choose the non-zero ones. Here we keep in
#             # mind the specified probability to ensure proper sampling
#             object_start_star_from_subject = generate_star_walks_object_centre(g, [start_point], 1, max_size_star,
#                                                                                [start_triple])
#             object_start_star_from_object = generate_star_walks_object_centre(g, [start_triple[2]], 1, max_size_star,
#                                                                               [start_triple])
#             subject_start_star_from_subject = generate_star_walks_subject_centre(g, [start_point], 1, max_size_star,
#                                                                                  [start_triple])
#             subject_start_star_from_object = generate_star_walks_subject_centre(g, [start_triple[2]], 1, max_size_star,
#                                                                                 [start_triple])
#             all_generated_stars = [object_start_star_from_object, object_start_star_from_subject,
#                                    subject_start_star_from_object, subject_start_star_from_subject]
#             # The unholiest of all codes to distribute probabilities of empty stars to non-empty stars
#             empty_stars = sum([1 for star in all_generated_stars if len(star) == 0])
#             probabilities = [p_object_star / 2, (1 - p_object_star) / 2, p_object_star / 2, (1 - p_object_star) / 2]
#             if empty_stars == 4:
#                 print("No star-path walk possible")
#                 continue
#             if empty_stars > 0:
#                 total_weight_to_zero_prob = sum([prob for i, prob in enumerate(probabilities) if
#                                                  len(all_generated_stars[i]) == 0])
#                 probabilities = [
#                     prob + (total_weight_to_zero_prob / (4 - empty_stars)) if len(all_generated_stars[i]) > 0 else 0
#                     for i, prob in enumerate(probabilities)]
#             start_star = random.choices(population=all_generated_stars, weights=probabilities)
#             print(start_star)
#
#
#     # TODO: Generation of complex walks that start with stars walks instead of path
#     pass


def generate_complex_walks(g, start_points, repeats, min_size_walk, max_size_walk):
    # Track how often predicates are used, so we down sample common predicates like friendOf
    predicate_usage_counts = {}
    all_walks = []
    # Iterate over start points
    for start_point in start_points:
        # Get all possible start triples from current start point
        possible_triples_from_start_point = list(g.triples((start_point, None, None)))
        possible_triples_from_start_point.extend(list(g.triples((None, None, start_point))))

        for i in range(repeats):
            # Randomly set this walks size
            walk_size = random.randint(min_size_walk, max_size_walk)

            # For n repeats, randomly select a start triple
            start_triple = random.choice(possible_triples_from_start_point)

            # Track a Set of all points we can extend from
            expandable_points = {start_triple[0], start_triple[1]}

            # Track a set of triples that are already in the walk, this will be used to prevent us going 'backward'
            # in a walk
            triples_in_walk = set()
            triples_in_walk.add(start_triple)

            n_triples_added = 0
            # While we haven't reached our max size, randomly expand some direction
            while n_triples_added < walk_size:
                # Get all triples that are possible from the open terms of the walk
                all_triples = set()
                for head in expandable_points:
                    # Update possible triples with all triples with head as subject
                    all_triples.update(set(g.triples((head, None, None))))
                    # Update possible triples with all triples with head as object
                    all_triples.update(set(g.triples((None, None, head))))

                # Remove triples currently in walk
                all_triples = all_triples.difference(triples_in_walk)

                # Convert to list to use in triple choosing
                all_triples = list(all_triples)

                # If we have no triples to select we break
                if len(all_triples) == 0:
                    break

                # Weighted random selection of triple to add to walk
                chosen_index, predicate_usage_counts = choose_triple_weighted(
                    all_triples, predicate_usage_counts)
                predicate_usage_counts = update_counts_predicates_used(
                    all_triples, chosen_index, predicate_usage_counts)

                # Update walk
                chosen_triple = all_triples[chosen_index]
                triples_in_walk.add(chosen_triple)

                # Update head, if we update head with term already in head it will simply not add it due to using a set
                expandable_points.add(chosen_triple[0])
                expandable_points.add(chosen_triple[2])

                # Update number of triples in walk
                n_triples_added += 1

            # Add complete walk to all walks list
            if len(triples_in_walk) >= min_size_walk:
                all_walks.append(list(triples_in_walk))
        break
    return all_walks


def generate_complex_queries(g, p_literal):
    all_subj = get_all_subject(g)

    complex_walks = generate_complex_walks(g, all_subj, 10, 3, 4)

    # Count predicates in walks
    predicate_counts = {}
    for walk in complex_walks:
        for triple in walk:
            if triple[1] not in predicate_counts:
                predicate_counts[triple[1]] = 1
                continue
            predicate_counts[triple[1]] += 1

    complex_queries = []
    for walk in complex_walks:
        walk_query = generate_complex_query_string(walk, p_literal)
        complex_queries.append(walk_query)
        break
    print("Filter complex queries")
    complex_queries = filter_isomorphic_queries(complex_queries)
    predicate_counts = count_predicates_queries(complex_queries)
    return complex_queries



def generate_complex_query_string(walk, p_literal):
    outer_triples = determine_outer_triples(walk)
    variable_counter = 1
    non_variable_edges = 0
    term_to_variable_dict = {}
    walk_query = "SELECT * WHERE { "
    for i, triple in enumerate(walk):
        # Handle logic for creation variables in the dictionary
        if triple[0] not in term_to_variable_dict:
            term_to_variable_dict[triple[0]] = "?v{}".format(variable_counter)
            variable_counter += 1

        r = random.uniform(0, 1)
        # If our triple is not an outer triple never allow a non-variable in query string
        if r > p_literal or triple not in outer_triples:
            if triple[2] not in term_to_variable_dict:
                term_to_variable_dict[triple[2]] = "?v{}".format(variable_counter)
                variable_counter += 1
            # Get r to max value = 1, so triple string will always be with variable
            triple_string, non_variable_edge = generate_triple_string(triple, 1, p_literal,
                                                                      term_to_variable_dict,
                                                                      object_centre=False)
        # Here we add literal as object
        else:
            triple_string, non_variable_edge = generate_triple_string(triple, r, p_literal,
                                                                      term_to_variable_dict,
                                                                      object_centre=False)
        if non_variable_edge:
            non_variable_edges += 1
        walk_query += triple_string
    walk_query += "}"
    return walk_query


def determine_outer_triples(walk):
    # Get all entities to get entities that occur multiple times
    all_entities_in_walk = []
    for triple in walk:
        all_entities_in_walk.append(triple[0])
        all_entities_in_walk.append(triple[2])

    all_subj_walk = set([triple[0] for triple in walk])
    all_obj_walk = set([triple[2] for triple in walk])

    # Get all objects that are not connected to another triple pattern
    outer_objects = all_obj_walk.difference(all_subj_walk)
    outer_objects_no_common = [obj for obj in outer_objects if all_entities_in_walk.count(obj) < 2]

    # Get associated triples, these triples can become literal queries in query generation
    outer_object_triples = set([triple for triple in walk if triple[2] in outer_objects_no_common])
    return outer_object_triples


def count_predicates_queries(queries):
    predicate_counts = dict()
    for query in queries:
        graph_of_query = convert_query_to_graph(query)
        for _, p, _ in graph_of_query:
            predicate_counts[p] = predicate_counts.get(p, 0) + 1

    return predicate_counts


def generate_path_star_walks(g, start_points, repeats, max_stars_in_walk, max_size_star, max_size_path, p_object_star):
    all_walks = []
    start_points_no_walk = []
    for start_point in start_points:
        # For each run we first generate a path query of random size. Then from this path query we randomly select
        # An entity and use that as the centre of our star query. The number of stars to add in the path is random.
        # Not sure how to do other way around stars yet
        for i in range(repeats):
            path_walk = generate_path_walks(g, [start_point], 1, max_size_path)
            if len(path_walk) == 0:
                # TODO: Generate from these points complex walks that start with either object or subject star (decided
                # TODO: randomly at first and if one doesnt work try the other), as some start points may point to an
                # TODO: Object that is only involved in a Star shape (City105 -> Country0, while country0 is only ever
                # TODO: An object and never subject thus walk cant be extended
                start_points_no_walk.append(start_point)
                # print("We found a start point with NO possible walks. \n"
                #       "Start point is: {}.".format(start_point))
                # generate_star_path_walks(g, [start_point], 1, max_stars_in_walk,
                #                          max_size_star, max_size_path, p_object_star)
                continue
            path_walk = path_walk[0]
            complex_walk = path_walk.copy()

            num_stars = random.randrange(1, max(2, max_stars_in_walk))
            possible_star_locations = path_walk
            stars_and_index = []
            for j in range(num_stars):
                star_index = random.randrange(len(possible_star_locations))

                if star_index == len(possible_star_locations) - 1 and random.randrange(0, 2) == 0 or star_index == 0:
                    # If we either are adding star at start subject of walk or at the end object of the walk (with
                    # Probability 0.5 we only have to ignore one triple when creating a star
                    to_ignore = [possible_star_locations[star_index]]
                else:
                    to_ignore = [possible_star_locations[star_index - 1], possible_star_locations[star_index]]

                # Generate either an object centred or subject centered star
                if random.uniform(0, 1) > p_object_star:
                    star_walk = generate_star_walks_object_centre(g, [possible_star_locations[star_index][2]],
                                                                  1, max_size_star, to_ignore)
                else:
                    star_walk = generate_star_walks_subject_centre(g, [possible_star_locations[star_index][0]],
                                                                   1, max_size_star, to_ignore)
                # If there are no star walks for a certain point we ignore the start point
                if star_walk:
                    stars_and_index.append([star_walk[0], star_index])
            # If we didnt find any stars to place we dont add the walk to all walks, as we only want complex shapes
            # TODO  Verify correct usage continue keyword
            if len(stars_and_index) == 0:
                continue

            for star in stars_and_index:
                complex_walk[star[1]:1] = star[0]
                # Remove the used subject from consideration for star generation
            all_walks.append(complex_walk)
    # Here we give all start subjects that couldnt generate a path walk a second try by trying to generate star walks
    # for start_point_no_walk in start_points_no_walk:
    #     start_triple = get_start_triple_walk(start_point_no_walk, g)
    #     if not start_triple:
    #         continue
    #     if random.uniform(0, 1) > p_object_star:
    #         star_walk = generate_star_walks_object_centre(g, start_triple[2],
    #                                                       1, max_size_star, [start_triple])
    #         if not star_walk:
    #             pass
    #
    #     else:
    #         star_walk = generate_star_walks_subject_centre(g, start_triple[0],
    #                                                        1, max_size_star, [start_triple])

    return all_walks, start_points_no_walk


# def generate_path_star_queries(g, repeats, max_stars_in_walk, max_size_star, max_size_path, p_object_star):
#     # TODO Determine / research if we should allow cycles in query graph, and what that means for the generated queries
#     # TODO Should we include literals in nodes
#     complex_queries = []
#     all_subj = get_all_subject(g)
#     walks, no_walk_start_points = generate_path_star_walks(g, all_subj, repeats, max_stars_in_walk, max_size_star,
#                                                            max_size_path, p_object_star)
#     for walk in walks:
#         variable_counter = 1
#         term_to_variable_dict = {}
#         n_triples = 0
#         for triple in walk:
#             if triple[0] not in term_to_variable_dict:
#                 term_to_variable_dict[triple[0]] = "?v{}".format(variable_counter)
#                 variable_counter += 1
#             if triple[2] not in term_to_variable_dict:
#                 term_to_variable_dict[triple[2]] = "?v{}".format(variable_counter)
#                 variable_counter += 1
#         walk_query = "SELECT * WHERE { "
#
#         for triple in walk:
#             triple_string = stringify_triple_pattern_complex(triple, term_to_variable_dict)
#             walk_query += triple_string
#         walk_query += "}"
#         complex_queries.append([walk_query, len(walk)])
#     complex_queries = filter_isomorphic_queries(complex_queries)
#
#     return complex_queries, no_walk_start_points


def stringify_triple_pattern_complex(triple, term_to_variable):
    triple_string = "{} <{}> {} . ".format(
        term_to_variable[triple[0]],
        triple[1],
        term_to_variable[triple[2]]
    )
    return triple_string


def main(save_location=None):
    instantiation_benchmark_location = r"C:\Users\Administrator\projects\benchmarks\watdiv-dataset\dataset.nt"
    input_graph = Graph()
    input_graph.parse(instantiation_benchmark_location, format="nt")
    generate_complex_queries(input_graph, .8)
    # generated_path_queries = generate_path_queries(input_graph, 1, 5, .25)
    # subject_stars, object_stars = generate_star_queries(input_graph, 1, 5, .25)
    # generated_complex_queries, no_walk_points = generate_path_star_queries(input_graph, 2, 2, 4, 4, .25)
    # print("generated {} queries".format(len(generated_complex_queries)))
    # print("No queries for {}/{} start points".format(len(set(no_walk_points)), len(get_all_subject(input_graph))))

    # all_queries = np.array(list(itertools.chain(subject_stars, object_stars, generated_path_queries)))
    # np.random.shuffle(all_queries)
    #
    # if save_location:
    #     np.save(save_location, all_queries)
    # print(subject_stars[0][0])
    # for query in all_queries:
    #     cardinality = 0
    #     for r in input_graph.query(subject_stars[0][0]):
    #         cardinality += 1
    #     break


if __name__ == "__main__":
    # main("output/queries/queries_star_path_iso.npy")
    main()
