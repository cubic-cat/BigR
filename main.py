"""Project entry point."""

from configs import embedding_config, llm_config, vector_db_config


def main() -> None:
    print("RAG project initialized.")
    print(f"Embedding model: {embedding_config.model_name}")
    print(f"LLM model: {llm_config.model_name}")
    print(f"Vector store path: {vector_db_config.storage_path}")


if __name__ == "__main__":
    main()
