from rag_eval.data_models import EvalContainer

# Example dataset for testing the RAG evaluator
test_examples = [
    EvalContainer(
        query="What is the capital of Italy and how many people live there?",
        ground_truth_answer="Italy's capital is Rome. Rome has about 2.8 million inhabitants in the city.",
        generated_answer="Italy's capital is Rome with 4 million inhabitants.",
        retrieved_texts=[
            "Rome is the capital city of Italy.",
            "The population of the Comune di Roma is approximately 2.8 million (not 4 million)."
        ],
    ),
    EvalContainer(
        query="Who wrote The Hobbit?",
        ground_truth_answer="The Hobbit was written by J.R.R. Tolkien.",
        generated_answer="The Hobbit was written by J.R.R. Tolkien in 1937.",
        retrieved_texts=[
            "The Hobbit is a fantasy novel by English author J.R.R. Tolkien.",
            "It was first published in 1937."
        ],
    ),
    EvalContainer(
        query="What is the tallest mountain in the world?",
        ground_truth_answer="Mount Everest is the tallest mountain in the world, standing at 8,849 meters.",
        generated_answer="Mount Everest is the tallest mountain in the world, standing at 8,848 meters.",
        retrieved_texts=[
            "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas.",
            "Its most recent official height measurement is 8,849 meters."
        ],
    ),
]
