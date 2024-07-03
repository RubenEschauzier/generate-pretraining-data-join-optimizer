import math
import random
from tqdm.auto import tqdm
from src.random_query_generation.utils import update_counts_predicates_used, choose_triple_weighted, get_all_subject, \
    get_all_object, generate_triple_string, track_equivalent_predicates, filter_equivalent_queries, \
    filter_isomorphic_queries, generate_corrupted_predicates_walks


def generate_star_queries(g, repeats, max_size_star, p_literal, p_walk_corrupt):
    all_subj = get_all_subject(g)
    all_obj = get_all_object(g)

    # Get all different star walks
    predicate_usage_counts_subject_centre = {}
    predicate_usage_counts_object_centre = {}

    print("Generating star walks subject centre \n")
    star_walks_subject_centre = generate_star_walks_subject_centre(g, all_subj, repeats, max_size_star,
                                                                   predicate_usage_counts_subject_centre)
    # Add walks that have random predicates in them to include randomly_generated_queries with result size = 0
    # TODO Add n_corrupted, max_corrupted, p_corruption as params
    print("Corrupting star walks subject centre \n")
    corrupted_predicates_walks_subject_centre = generate_corrupted_predicates_walks(g, star_walks_subject_centre,
                                                                                    1, 2, p_walk_corrupt)
    star_walks_subject_centre.extend(corrupted_predicates_walks_subject_centre)

    print("Generating star walks object centre \n")
    star_walks_object_centre = generate_star_walks_object_centre(g, all_obj, int(math.floor(repeats / 2)),
                                                                 max_size_star,
                                                                 predicate_usage_counts_object_centre)
    # Add walks that have random predicates in them to include randomly_generated_queries with result size = 0
    # TODO Add n_corrupted, max_corrupted, p_corruption as params
    print("Corrupting star walks object centre \n")
    corrupted_predicates_walks_object_centre = generate_corrupted_predicates_walks(g, star_walks_object_centre,
                                                                                   1, 2, p_walk_corrupt)
    star_walks_object_centre.extend(corrupted_predicates_walks_object_centre)

    queries_subject_centre = []
    queries_object_centre = []
    used_predicates_subject_centre = set()
    used_predicates_object_centre = set()
    # Generate subject_centre randomly_generated_queries
    print("Generating subject centre star randomly_generated_queries from walks \n")
    for walk_subject_centre in tqdm(star_walks_subject_centre):
        predicates = tuple(sorted([triple[1] for triple in walk_subject_centre]))
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
            if r > p_literal:
                if triple[2] not in term_to_variable_dict:
                    term_to_variable_dict[triple[2]] = "?v{}".format(variable_counter)
                    variable_counter += 1
            triple_string, non_variable_edge = generate_triple_string(g, all_subj, all_obj, triple, r, p_literal,
                                                                      term_to_variable_dict, .3)

            if non_variable_edge:
                non_variable_edges += 1
            walk_query += triple_string
        walk_query += "}"

        queries_subject_centre.append(walk_query)

    print("Filter subject centre star randomly_generated_queries \n")
    queries_subject_centre = filter_isomorphic_queries(queries_subject_centre)

    print("Generating object centre star randomly_generated_queries from walks \n")
    for walk_obj_centre in tqdm(star_walks_object_centre):
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
            triple_string, non_variable_edge = generate_triple_string(g, all_subj, all_obj, triple, 1, p_literal,
                                                                      term_to_variable_dict, .3,
                                                                      object_centre=True)
            if non_variable_edge:
                non_variable_edges += 1

            walk_query += triple_string
        walk_query += "}"
        queries_object_centre.append(walk_query)

    print("Filtering object centre star randomly_generated_queries \n")
    queries_object_centre = filter_isomorphic_queries(queries_object_centre)

    return queries_subject_centre, queries_object_centre


def generate_star_walks_subject_centre(g, start_points, repeats, max_points,
                                       predicate_usage_counts, triples_to_ignore=None):
    all_walks = []
    for star_centre in tqdm(start_points):
        for i in range(repeats):
            possible_star_points = get_possible_star_points_subject_centre(g, star_centre, triples_to_ignore)
            all_walks = extend_star_walk(possible_star_points, max_points, all_walks,
                                         predicate_usage_counts)
    return all_walks


def generate_star_walks_object_centre(g, start_points, repeats, max_points,
                                      predicate_usage_counts, triples_to_ignore=None):
    all_walks = []
    for star_centre in tqdm(start_points):
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


# TODO: Docstring, note triple_to_ignore denotes the triple that connects multiple star randomly_generated_queries
#  together
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
