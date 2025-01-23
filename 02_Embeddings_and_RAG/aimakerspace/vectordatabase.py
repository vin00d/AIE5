import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Literal
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the negative Euclidean distance between two vectors.
    Returns negative distance so that higher values (closer to 0) indicate more similarity,
    maintaining consistency with cosine_similarity where higher values mean more similar."""
    return -np.linalg.norm(vector_a - vector_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None, distance_metric: Literal["cosine", "euclidean"] = "cosine"):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.distance_metrics = {
            "cosine": cosine_similarity,
            "euclidean": euclidean_distance
        }
        if distance_metric not in self.distance_metrics:
            raise ValueError(f"Distance metric must be one of {list(self.distance_metrics.keys())}")
        self.default_distance_metric = self.distance_metrics[distance_metric]

    def insert(self, key: str, vector: np.array) -> None:
        self.vectors[key] = vector

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = None,
    ) -> List[Tuple[str, float]]:
        if distance_measure is None:
            distance_measure = self.default_distance_metric
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = None,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    # Test with cosine similarity
    vector_db_cosine = VectorDatabase(distance_metric="cosine")
    vector_db_cosine = asyncio.run(vector_db_cosine.abuild_from_list(list_of_text))
    k = 2

    # Test with Euclidean distance
    vector_db_euclidean = VectorDatabase(distance_metric="euclidean")
    vector_db_euclidean = asyncio.run(vector_db_euclidean.abuild_from_list(list_of_text))

    query = "I think fruit is awesome!"
    
    # Compare results from both distance metrics
    cosine_results = vector_db_cosine.search_by_text(query, k=k)
    print(f"Cosine similarity - Closest {k} matches:", cosine_results)
    
    euclidean_results = vector_db_euclidean.search_by_text(query, k=k)
    print(f"Euclidean distance - Closest {k} matches:", euclidean_results)
