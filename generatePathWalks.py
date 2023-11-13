import rdflib.term
from rdflib.graph import Graph
import random


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
def extend_walk_path(g, head, tail, walk):
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
                head, tail, walk = extend_walk_path(g, head, tail, walk)

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
                    # If literal at end we don't add brackets, because not URI
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
    return queries


def generate_star_queries(g, prob_non_variable_edge):
    all_subj = get_all_subject(g)
    all_obj = get_all_object(g)

    # Get all different star walks
    star_walks_subject_centre = generate_star_walks_subject_centre(g, all_subj, 10, 6)
    star_walks_object_centre = generate_star_walks_object_centre(g, all_obj, 5, 6)

    queries_subject_centre = []
    queries_object_centre = []

    # Generate subject_centre queries
    for walk_subject_centre in star_walks_subject_centre:
        variable_counter = 1
        term_to_variable_dict = {}
        walk_query = "CONSTRUCT WHERE { "
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
            triple_string = generate_triple_string(triple, r, prob_non_variable_edge, term_to_variable_dict,
                                                   object_centre=False)
            walk_query += triple_string
        walk_query += "}"
        queries_subject_centre.append([walk_query, len(walk_subject_centre)])

    for walk_obj_centre in star_walks_object_centre:
        variable_counter = 1
        term_to_variable_dict = {}
        walk_query = "CONSTRUCT WHERE { "
        for i, triple in enumerate(walk_obj_centre):
            # Handle logic for creation variables in the dictionary, object centres are only variables
            if triple[2] not in term_to_variable_dict:
                term_to_variable_dict[triple[2]] = "?v{}".format(variable_counter)
                variable_counter += 1
            if triple[0] not in term_to_variable_dict:
                term_to_variable_dict[triple[0]] = "?v{}".format(variable_counter)
                variable_counter += 1
            # Here we set r to arbitrary one, indicating that it will never have literals in the query
            triple_string = generate_triple_string(triple, 1, prob_non_variable_edge, term_to_variable_dict,
                                                   object_centre=True)
            walk_query += triple_string
        walk_query += "}"
        queries_object_centre.append([walk_query, len(walk_obj_centre)])
    return queries_subject_centre


# Separate the logic for variable dict and string creation
def generate_triple_string(triple, r, prob_non_variable_edge, term_to_variable_dict, object_centre):
    if r > prob_non_variable_edge or object_centre:
        triple_string = "{} <{}> {} . ".format(
            term_to_variable_dict[triple[0]],
            triple[1],
            term_to_variable_dict[triple[2]]
        )
    else:
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
    return triple_string


def generate_star_walks_subject_centre(g, start_points, repeats, max_points, triples_to_ignore=None):
    all_walks = []
    for star_centre in start_points:
        for i in range(repeats):
            possible_star_points = get_possible_star_points_subject_centre(g, star_centre, triples_to_ignore)
            all_walks = extend_star_walk(possible_star_points, max_points, all_walks)
    return all_walks


def generate_star_walks_object_centre(g, start_points, repeats, max_points, triples_to_ignore=None):
    all_walks = []
    for star_centre in start_points:
        for i in range(repeats):
            possible_star_points = get_possible_star_points_object_centre(g, star_centre, triples_to_ignore)
            all_walks = extend_star_walk(possible_star_points, max_points, all_walks)
    return all_walks


def extend_star_walk(possible_star_points, max_points, all_walks):
    walk = []
    size = random.randrange(2, max_points)

    for j in range(size):
        if len(possible_star_points) == 0:
            break
        new_star_point, possible_star_points = sample_new_star_point(possible_star_points)
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


def sample_new_star_point(triples_to_sample):
    chosen_index = random.randrange(len(triples_to_sample))
    returned_triple = triples_to_sample[chosen_index]
    triples_to_sample.remove(returned_triple)
    return returned_triple, triples_to_sample


def generate_star_path_walks(g, start_points, repeats, max_stars_in_walk, max_size_star, max_size_path, p_object_star):
    # 4 options for starting a walk from a star. You either make object star with subject or subject star with subject.
    # Then, if that doesn't work you try to make an object star with object or subject star with object
    # For continuing the walk, we can't start from our start_point, so we should choose either a random subject
    # from an object star, or choose a random object from a subject star. In all case we should check whether we
    # actually created a valid complex query with at least a path and star
    all_walks = []
    for start_point in start_points:

        for i in range(repeats):
            start_triple = get_start_triple_walk(start_point)
            r = random.uniform(0, 1)
            generate_backup_subject_star = False
            # For simplicity, we simply generate all star walks we can and choose the non-zero ones. Here we keep in mind
            # the specified probability to ensure proper sampling
            object_start_star_from_subject = generate_star_walks_object_centre(g, [start_point], 1, max_size_star,
                                                                         [start_triple])
            object_start_star_from_object = generate_star_walks_object_centre(g, [start_triple[2]], 1, max_size_star,
                                                                        [start_triple])
            subject_start_star_from_subject = generate_star_walks_subject_centre(g, [start_point], 1, max_size_star,
                                                                         [start_triple])
            subject_start_star_from_object = generate_star_walks_subject_centre(g, [start_triple[2]], 1, max_size_star,
                                                                        [start_triple])
            all_generated_stars = [object_start_star_from_object, object_start_star_from_subject,
                                subject_start_star_from_object, subject_start_star_from_subject]
            options = [x for x in all_generated_stars if len(x)>0]



            if r > p_object_star:
                # Generate subject star from both subject and object, inefficient but more readable
                start_star_from_subject = generate_star_walks_subject_centre(g, [start_point], 1, max_size_star, [start_triple])
                start_star_from_object = generate_star_walks_subject_centre(g, [start_triple[2]], 1, max_size_star, [start_triple])
                # Randomly choose if both exist
                if len(start_star_from_object) != 0 and len(start_star_from_subject != 0):
                    r = random.randrange(0,2)
                    if r == 0:
                        start_star = start_star_from_subject
                    else:
                        start_star = start_star_from_object
                # If one or the other doesn't exist pick that one (simple)
                if len(start_star_from_object) == 0 and len(start_star_from_subject != 0):
                    start_star = start_star_from_subject
                if len(start_star_from_object) != 0 and len(start_star_from_subject == 0):
                    start_star = start_star_from_object

                # If both stars don't exist try again but this time a subject star
                if len(start_star_from_object) == 0 and start_star_from_subject == 0:
                    generate_backup_subject_star = True
            if r < p_object_star or generate_backup_subject_star:


    # TODO: Generation of complex walks that start with stars walks instead of path
    pass


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
                print("We found a start point with NO possible walks. \n"
                      "Start point is: {}.".format(start_point))
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
    for start_point_no_walk in start_points_no_walk:
        start_triple = get_start_triple_walk(start_point_no_walk, g)
        if not start_triple:
            continue
        if random.uniform(0, 1) > p_object_star:
            star_walk = generate_star_walks_object_centre(g, start_triple[2],
                                                          1, max_size_star, [start_triple])
            if not star_walk:
                pass

        else:
            star_walk = generate_star_walks_subject_centre(g, start_triple[0],
                                                           1, max_size_star, [start_triple])

    return all_walks


def generate_path_star_queries(g, repeats, max_stars_in_walk, max_size_star, max_size_path, p_object_star):
    # TODO Determine / research if we should allow cycles in query graph, and what that means for the generated queries
    # TODO Should we include literals in nodes
    complex_queries = []
    all_subj = get_all_subject(g)
    walks = generate_path_star_walks(g, all_subj, repeats, max_stars_in_walk, max_size_star, max_size_path,
                                     p_object_star)
    for walk in walks:
        variable_counter = 1
        term_to_variable_dict = {}
        for triple in walk:
            if triple[0] not in term_to_variable_dict:
                term_to_variable_dict[triple[0]] = "?v{}".format(variable_counter)
                variable_counter += 1
            if triple[2] not in term_to_variable_dict:
                term_to_variable_dict[triple[2]] = "?v{}".format(variable_counter)
                variable_counter += 1
        walk_query = "CONSTRUCT WHERE { "

        for triple in walk:
            triple_string = stringify_triple_pattern_complex(triple, term_to_variable_dict)
            walk_query += triple_string
        walk_query += "}"
        complex_queries.append(walk_query)
    return complex_queries


def stringify_triple_pattern_complex(triple, term_to_variable):
    triple_string = "{} <{}> {} . ".format(
        term_to_variable[triple[0]],
        triple[1],
        term_to_variable[triple[2]]
    )
    return triple_string


def delete_duplicate_queries(query_list):
    # TODO Determine form of query equivalence / find library that does this to truly delete duplicates
    return list(set(query_list))


if __name__ == "__main__":
    input_graph = Graph()
    input_graph.parse("dataWatDiv/dataset.nt", format="nt")
    # generate_star_queries(input_graph, .25)
    generated_complex_queries = generate_path_star_queries(input_graph, 5, 2, 4, 4, .25)
    print("generated {} queries".format(len(generated_complex_queries)))
    non_duplicate_queries = delete_duplicate_queries(generated_complex_queries)
    print("Removed duplicates, now {} queries".format(len(generated_complex_queries)))

    # generated_path_queries = generate_path_queries(input_graph, .25)
    # print(generated_path_queries)
    # num_triples = 0
    # for r in input_graph.query(generated_path_queries[0][0]):
    #     num_triples += 1
    # print(num_triples)
    pass
