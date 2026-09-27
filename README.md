# Menu Name Classifier

This project guesses a dish's **ingredient list from its menu name alone**.
For example, you give it "BBQ Chicken Bowl" and it returns a likely set of
ingredients. It uses a classic bag-of-words pipeline trained on
[RecipeNLG](https://recipenlg.cs.put.poznan.pl/), a collection of about 2.2
million recipes.

It was built in 2023 as a first baseline for a foodservice analytics project.
Later versions of the same idea moved to LLMs (see `dfl-evaluation` and
`llm-eval-binclass`).

## How it works

`newrecipe.py` runs the whole pipeline from top to bottom:

1. **Load** 800,000 random recipes from `Inputs/full_dataset.csv` and keep
   only the `title` and `ingredients` columns.
2. **Vectorize** the titles with a binary `CountVectorizer`, which only
   records whether each word appears, not how often.
3. **Shrink** the result to 100 features with `TruncatedSVD`.
4. **Train** a `BernoulliNB` model in batches of 2,500 rows with
   `partial_fit`, keeping 20% of the data aside for testing. Each distinct
   ingredient list counts as one class.
5. **Evaluate** with accuracy and micro-averaged precision, recall, and F1.
6. **Save** `model.pkl`, `vectorizer.joblib`, and `svd_model.joblib`, then run
   a few example predictions.

At the bottom of the script, there's a short example of how to load the saved
files and call `predict(dish_name)`.

## Limitations

This is a baseline, and the way the problem is set up holds it back:

- **Whole ingredient lists are the classes.** Almost every recipe has its own
  unique list, so the model mostly finds the closest recipe it has seen and
  returns that recipe's list. It doesn't build a new one. Treating each
  ingredient as its own yes/no output (multi-label classification) would fit
  the task better.
- **Exact-match metrics don't tell the full story** for the same reason. A
  prediction with nine out of ten ingredients right scores the same as one
  with none right.
- **The model is huge.** The unzipped `model.pkl` is about 28 GB, because
  BernoulliNB stores numbers for every class, and there are hundreds of
  thousands of them.
- Squeezing short titles into 100 features throws away most of the
  vocabulary.

## Getting the data

The data and model files aren't in git. The training data is the public
RecipeNLG dataset, and the model files are rebuilt when you run the script.

1. Download the RecipeNLG dataset from the
   [RecipeNLG site](https://recipenlg.cs.put.poznan.pl/).
2. Unzip it so the file sits at `Inputs/full_dataset.csv`.

## Running

The script needs Python 3.8 or newer.

```bash
pip install pandas numpy scikit-learn joblib tqdm
```

```bash
python newrecipe.py
```

Training needs a lot of RAM. The biggest spikes come from loading 800,000
rows and from the SVD step. When it finishes, the script writes `model.pkl`,
`vectorizer.joblib`, and `svd_model.joblib` to the project folder.

## Files

| Path | Contents |
| --- | --- |
| `newrecipe.py` | Training, evaluation, and example predictions |
| `Inputs/full_dataset.csv` | RecipeNLG data (2.2 GB, not in git) |
| `model.pkl`, `*.joblib` | Trained model, vectorizer, and SVD (not in git) |
