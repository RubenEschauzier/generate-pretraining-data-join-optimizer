import os

from src.entity_embedding_generation.onehot_entity_embedding import main_onehot_embedding
from src.entity_embedding_generation.rdf2vec_entity_embedding import main_rdf2vec

if __name__ == "__main__":
    # This is a system-wide instantiation of watdiv used for every project. This is to prevent mixing watdiv
    # instantiations and queries causing invalid benchmark results. For use outside of my (Ruben's) computer change this
    instantiation_benchmark_location = r"C:\Users\Administrator\projects\benchmarks\watdiv-dataset\dataset.nt"

    project_root = os.path.join(os.getcwd(), "..", "..")
    onehot_embedding_save_location = os.path.join(project_root, "output", "entity_embeddings",
                                                  "embeddings_onehot_encoded.txt")
    rdf2vec_vector_save_location = os.path.join(project_root, "output", "entity_embeddings",
                                                r"\rdf2vec_vectors_depth_{}_test.txt")
    onehot_embed = False
    rdf2vec_embed = True

    if onehot_embed:
        main_onehot_embedding(instantiation_benchmark_location, onehot_embedding_save_location)

    if rdf2vec_embed:
        n_sim_pred = 1000
        n_sim_subj = 200
        n_sim_obj = 200
        depth_walk = 1
        main_rdf2vec(n_sim_pred, n_sim_subj, n_sim_obj, depth_walk,
                     instantiation_benchmark_location, onehot_embedding_save_location)