# interest-memory

Learns what a reader likes from weighted feedback and scores new text by it. Uses local embeddings (fastembed) and SQLite. Makes no LLM calls.

## Usage

1. Install it in your project: `uv add git+https://github.com/sjson421/interest-memory`
2. Create an `InterestMemory` with a path to a SQLite file. The file is created if it does not exist.
3. Add seed phrases for topics you already like with `add_seeds`. Seeds never fade.
4. Each time a reader reacts to an item, call `record` with an item ID, its text, a weight, and the time. Use a positive weight for liked items and a negative weight for disliked ones. A bigger number means a stronger signal. Call `forget` with the ID to undo it.
5. Call `score` with a list of new texts and the current time. It returns one number per text. Higher means closer to the reader's interests.
6. Optional: save a written summary of the reader's interests with `set_profile`, and read the latest one with `profile`.

The first call that needs embeddings downloads the fastembed model, about 70 MB.

To work on this repo: `uv sync`, then `uv run pytest`.

## Scoring

1. Embed the new text as a vector, so texts with similar meanings land close together.
2. Compare it with every stored item, both seeds and feedback. Similarity runs from 0 (unrelated) to 1 (same meaning). Negative similarity counts as 0.
3. Keep the `k` most similar items (default 10).
4. Decay each item's weight by half every `half_life_days` (default 60). Seeds never decay.
5. Multiply similarity by decayed weight for each item, add the results, and divide by `k`.

The divisor is always `k`, so scores stay small while there is little feedback.

