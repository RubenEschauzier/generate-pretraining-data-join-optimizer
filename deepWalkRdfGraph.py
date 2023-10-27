from rdflib.graph import Graph
from gensim.models import Word2Vec
import itertools
import random

def get_all_predicates(g):
    all_predicate_occurrence = []
    predicates = set()
    entities = set()
    subjects = set()
    objects = set()
    predicates_uri = set()

    for (subj, pred, obj) in g:
        predicates.add(str(pred))
        predicates_uri.add(pred)
        entities.add(subj)
        entities.add(obj)
        subjects.add(subj)
        objects.add(obj)
        all_predicate_occurrence.append(pred)

    predicates = list(predicates)
    predicates_uri = list(predicates_uri)
    return predicates, predicates_uri, entities, all_predicate_occurrence, subjects, objects


def get_all_subj_obj(g):
    subj_dict = {}
    obj_dict = {}
    for (subj, pred, obj) in g:
        if subj in subj_dict:
            subj_dict[subj] += 1
        else:
            subj_dict[subj] = 1
        if obj in obj_dict:
            obj_dict[obj] += 1
        else:
            obj_dict[obj] = 1
    return subj_dict, obj_dict


def load_graph():
    g = Graph()
    g.parse("dataWatDiv/dataset.nt", format="nt")
    return g


def generate_walks(g, predicatesUri, num_sim_pred, depth_walk):
    all_walks = []
    num_pred_done = 0
    for predicate in predicatesUri:
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
            start_triple = tp[index]
            start_subject = start_triple[0]
            # This is the triple that the subject belongs to, for ordering
            start_object = start_triple[2]
            # This is the triple that the object belongs to, for ordering
            walk_in_progress.extend(list(start_triple))

            while num_triples_chosen < depth_walk:
                # Get all triples with starting object as subject
                start_object_as_subject = list(g.triples((start_object, None, None)))
                # All triples with starting subject as object
                start_subject_as_object = list(g.triples((None, None, start_subject)))
                # Merge all possible triples
                possible_triples = list(itertools.chain(start_object_as_subject, start_subject_as_object))
                if len(possible_triples) == 0:
                    break
                # Get new triple to walk to
                new_index = random.randrange(len(possible_triples))
                start_triple = possible_triples[new_index]
                if new_index < len(start_object_as_subject):
                    start_object = start_triple[2]
                    walk_in_progress.extend(list(start_triple))
                else:
                    extended_walk = list(start_triple)
                    start_subject = start_triple[0]
                    extended_walk.extend(walk_in_progress)
                    walk_in_progress = extended_walk

                # walk_in_progress.extend(list(start_triple))
                num_triples_chosen += 1

            if len(walk_in_progress) == depth_walk * 3 + 3:
                # Ugliest way to remove duplicate entries from walking, should do it in walk generation
                last_element = walk_in_progress[-1]
                del walk_in_progress[2::3]
                walk_in_progress.append(last_element)
                all_walks.append(walk_in_progress)
                num_walks_predicate += 1
    return all_walks


def generate_walks_from_subj(g, subjUri, num_sim_subj, depth_walk):
    all_walks = []
    num_subj_done = 0
    for subj in subjUri:
        print("Finish subj {}/{}".format(num_subj_done+1, len(subjUri)))
        num_subj_done += 1
        triples_with_subj = g.triples((subj, None, None))
        triples_with_subj_as_obj = g.triples((None, None, subj))
        tp = list(triples_with_subj)
        tp.extend(list(triples_with_subj_as_obj))
        num_walks_subj = 0
        for i in range(num_sim_subj):
            num_triples_chosen = 0
            walk_in_progress = []
            # Choose random triple from our list of triples
            index = random.randrange(len(tp))
            start_triple = tp[index]
            start_subject = start_triple[0]
            # This is the triple that the subject belongs to, for ordering
            start_object = start_triple[2]
            # This is the triple that the object belongs to, for ordering
            walk_in_progress.extend(list(start_triple))

            while num_triples_chosen < depth_walk:
                # Get all triples with starting object as subject
                start_object_as_subject = list(g.triples((start_object, None, None)))
                # All triples with starting subject as object
                start_subject_as_object = list(g.triples((None, None, start_subject)))
                # Merge all possible triples
                possible_triples = list(itertools.chain(start_object_as_subject, start_subject_as_object))
                if len(possible_triples) == 0:
                    break
                # Get new triple to walk to
                new_index = random.randrange(len(possible_triples))
                start_triple = possible_triples[new_index]
                if new_index < len(start_object_as_subject):
                    start_object = start_triple[2]
                    walk_in_progress.extend(list(start_triple))
                else:
                    extended_walk = list(start_triple)
                    start_subject = start_triple[0]
                    extended_walk.extend(walk_in_progress)
                    walk_in_progress = extended_walk

                num_triples_chosen += 1

            if len(walk_in_progress) == depth_walk * 3 + 3:
                # Ugliest way to remove duplicate entries from walking, should do it in walk generation
                last_element = walk_in_progress[-1]
                del walk_in_progress[2::3]
                walk_in_progress.append(last_element)
                all_walks.append(walk_in_progress)
                num_walks_subj += 1
    return all_walks


def get_all_subj_obj_in_walks(g, walks):
    subj_dict, obj_dict = get_all_subj_obj(g)
    subj_dict_walk = {}
    obj_dict_walk = {}
    for walk in walks:
        subjects = walk[0:-1:2]
        objects = walk[2::2]
        for subj in subjects:
            if subj in subj_dict_walk:
                subj_dict_walk[subj] += 1
            else:
                subj_dict_walk[subj] = 1
        for obj in objects:
            if obj in obj_dict_walk:
                obj_dict_walk[obj] += 1
            else:
                obj_dict_walk[obj] = 1
    return subj_dict_walk, obj_dict_walk, subj_dict, obj_dict


def get_average_difference_occurrences(subj_dict_walk, subj_dict, obj_dict_walk, obj_dict):
    total_difference_subj = 0
    total_missing_subj = 0
    for key, value in subj_dict.items():
        diff = value
        if key in subj_dict_walk:
            diff = value - subj_dict_walk[key]
        else:
            total_missing_subj += 1
        total_difference_subj += diff
    avg_diff_subj = total_difference_subj / len(subj_dict.items())

    total_difference_obj = 0
    total_missing_obj = 0
    for key, value in obj_dict.items():
        diff = value
        if key in obj_dict_walk:
            diff = value - obj_dict_walk[key]

def train_model(walks):
    corpus = [[str(word) for word in walk] for walk in walks]
    vector_dim = 128
    model = Word2Vec(corpus, min_count=1, window=5, vector_size=vector_dim, epochs=100)
    return model


def save_rdf_predicate_model(model, predicates):
    vector_dim = 128
    with open('dataWatDiv/vectorsWatDiv/vectors_depth_2.txt', 'w') as f:
        # First write predicates to file
        for predicate in predicates:
            if predicate in model.wv:
                to_write = str(predicate) + '[sep]' + ' '.join([str(x) for x in model.wv[predicate]])
                f.write(to_write)
                f.write('\n')
                pass
            else:
                to_write = str(predicate) + '[sep]' + ' '.join(['0' for _ in range(vector_dim)])
                f.write(to_write)
                f.write('\n')
        # Write non-predicates to file
        for key in model.wv.index_to_key:
            if key not in predicates:
                to_write = str(key) + '[sep]' + ' '.join([str(x) for x in model.wv[key]])
                f.write(to_write)
                f.write('\n')


def start_rdf_predicate_2_vec(save_model=True):
    num_sim_pred = 1000
    num_sim_subj = 10
    num_sim_obj = 10
    depth_walk = 1
    g = load_graph()
    predicates, predicates_uri, entities, all_pred_occurrences, subjects, objects = get_all_predicates(g)
    walk_subj = generate_walks_from_subj(g, subjects, num_sim_subj, depth_walk)
    walk_obj = generate_walks_from_subj(g, objects, num_sim_obj, depth_walk)
    walks = generate_walks(g, predicates_uri, num_sim_pred, depth_walk)
    combined_walks = []
    for walk_p in walks:
        combined_walks.append(walk_p)
    for walk in walk_subj:
        combined_walks.append(walk)
    for walk_o in walk_obj:
        combined_walks.append(walk_o)
    # subj_dict_walk, obj_dict_walk, subj_dict, obj_dict = get_all_subj_obj_in_walks(g, walks)
    # subj_dict_walk_c, obj_dict_walk_c, subj_dict_c, obj_dict_c = get_all_subj_obj_in_walks(g, combined_walks)
    #
    # get_average_difference_occurrences(subj_dict_walk, subj_dict, obj_dict_walk, obj_dict)
    # get_average_difference_occurrences(subj_dict_walk_c, subj_dict, obj_dict_walk_c, obj_dict)
    model = train_model(walks)
    if save_model:
        save_rdf_predicate_model(model, predicates)


if __name__ == "__main__":
    start_rdf_predicate_2_vec(False)
