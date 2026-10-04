import os
import json
import numpy as np
import faiss


class VectorStore:

    def __init__(self):

        os.makedirs("data", exist_ok=True)

        self.metadata_file = "data/vector_metadata.json"
        self.index_file = "data/vector.index"

        self.documents = []

        if os.path.exists(self.metadata_file):

            with open(
                self.metadata_file,
                "r",
                encoding="utf-8"
            ) as file:

                self.documents = json.load(file)

        self.index = None

        if os.path.exists(self.index_file):

            self.index = faiss.read_index(
                self.index_file
            )

    def save(self):

        with open(
            self.metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.documents,
                file,
                ensure_ascii=False
            )

        if self.index is not None:

            faiss.write_index(
                self.index,
                self.index_file
            )

    def add_documents(
        self,
        chunks,
        embeddings,
        branch,
        semester,
        subject,
        filename
    ):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        # Remove previous copy of this exact PDF
        self.documents = [
            doc
            for doc in self.documents
            if not (
                doc["metadata"]["filename"] == filename
                and doc["metadata"]["branch"] == branch
                and doc["metadata"]["semester"] == semester
                and doc["metadata"]["subject"] == subject
            )
        ]

        for i, chunk in enumerate(chunks):

            self.documents.append({

                "text": chunk["text"],

                "metadata": {
                    "page": chunk["page"],
                    "branch": branch,
                    "semester": semester,
                    "subject": subject,
                    "filename": filename
                },

                "embedding": embeddings[i].tolist()
            })

        # Rebuild FAISS index
        all_vectors = [
            document["embedding"]
            for document in self.documents
        ]

        if all_vectors:

            matrix = np.asarray(
                all_vectors,
                dtype="float32"
            )

            dimension = matrix.shape[1]

            self.index = faiss.IndexFlatIP(
                dimension
            )

            self.index.add(matrix)

        self.save()

    def search(
        self,
        query_embedding,
        branch,
        semester,
        subject,
        filename=None,
        top_k=5
    ):

        if self.index is None:

            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        query = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        # Search more candidates first,
        # then apply metadata filtering.
        search_count = min(
            top_k * 10,
            len(self.documents)
        )

        scores, indices = self.index.search(
            query,
            search_count
        )

        selected_documents = []
        selected_metadata = []
        selected_distances = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            document = self.documents[index]

            metadata = document["metadata"]

            if metadata["branch"] != branch:
                continue

            if metadata["semester"] != semester:
                continue

            if metadata["subject"] != subject:
                continue

            if filename is not None:

                if metadata["filename"] != filename:
                    continue

            selected_documents.append(
                document["text"]
            )

            selected_metadata.append(
                metadata
            )

            selected_distances.append(
                float(1 - score)
            )

            if len(selected_documents) >= top_k:
                break

        return {
            "documents": [
                selected_documents
            ],
            "metadatas": [
                selected_metadata
            ],
            "distances": [
                selected_distances
            ]
        }