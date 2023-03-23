import pickle
import os
# from sentence_transformers import SentenceTransformer, util


__file_path  = os.path.abspath(__file__)
__parent_dir = os.path.dirname(__file_path)
__model_path = "models/paraphrase-MiniLM-L6-v2.pkl"

__loaded_model = None
__doc_embeddings = None

def load_model_pickle(func):
    """
    A decorator function that loads a pickled model file into memory
    before executing the decorated function.deact

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """
    def inner( *args,**kwargs ):
        global __loaded_model
        if __loaded_model is None:
            model_file = "{}/{}".format(__parent_dir,__model_path)
            with open(model_file, 'rb') as f:
                __loaded_model = pickle.load(f)
        return func(*args,**kwargs)
    return inner



@load_model_pickle
def vectorize():
    """
    A function that performs vectorization on a given input using a
    pre-trained model.
    """
    print(__loaded_model)



@load_model_pickle
def fetch_top_k(query_embedding):
    """
    A function that fetches the top k most similar items to a given input
    using a pre-trained model.
    """

    



if __name__ == "__main__":
    vectorize()
