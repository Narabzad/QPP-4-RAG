from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer

# Create a sample request
query = Query(qid="1", text="What are the main features of Python?")
documents = [
    Document(
        docid="1",
        segment="""Python is a high-level programming language known for its 
        simplicity and readability. It supports multiple programming paradigms 
        including procedural, object-oriented, and functional programming."""
    ),
    Document(
        docid="2",
        segment="""Python was created by Guido van Rossum in 1991."""
    ),
    Document(
        docid="3",
        segment="""Python is widely used in web development, data analysis, 
        artificial intelligence, and scientific computing."""
    ),
]
request = Request(query=query, documents=documents)

# Option 1: Single model for all components
nuggetizer = Nuggetizer(model="gpt-4o")  # Uses same model for all components

# Option 2: Different models for each component
nuggetizer_mixed = Nuggetizer(
    creator_model="gpt-4o",  # Model for nugget creation
    scorer_model="gpt-3.5-turbo",  # Model for nugget scoring
    assigner_model="gpt-4o"  # Model for nugget assignment
)

# Create and score nuggets
scored_nuggets = nuggetizer.create(request)

# Print nuggets and their importance
for nugget in scored_nuggets:
    print(f"Nugget: {nugget.text}")
    print(f"Importance: {nugget.importance}\n")

# Assign nuggets to a specific document
assigned_nuggets = nuggetizer.assign(query.text, documents[0].segment, scored_nuggets)

# Print assignments
for nugget in assigned_nuggets:
    print(f"Nugget: {nugget.text}")
    print(f"Importance: {nugget.importance}")
    print(f"Assignment: {nugget.assignment}\n")