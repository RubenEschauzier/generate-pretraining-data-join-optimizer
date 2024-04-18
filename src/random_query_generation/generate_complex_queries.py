import random

from src.random_query_generation.utils import generate_triple_string, filter_isomorphic_queries, \
    count_predicates_queries, get_all_subject, update_counts_predicates_used, choose_triple_weighted


def generate_complex_queries(g, repeats, min_size_walk, max_size_walk, p_literal):
    all_subj = get_all_subject(g)

    complex_walks = generate_complex_walks(g, all_subj, repeats, min_size_walk, max_size_walk)

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

    print("Filter complex queries")
    complex_queries = filter_isomorphic_queries(complex_queries)
    predicate_counts = count_predicates_queries(complex_queries)
    return complex_queries


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
    return all_walks


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
