# What is this?

This is a tool that can be used to filter a large wordlist down to the "best" subset of words. Perfect for building 1d100 lists for oracles / inspirational prompts for writing / RPGs.

# What is this not?

This tool cannot create wordlists from nothing, it can only take large wordlists and make them smaller. It will only output words that were in your input list.

# How to use?

## Python

1) Download the [model](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip) file, unzip it, put it inside the `/model` directory
2) `pip install -r requirements.txt`
3) `python3 -c "import nltk; nltk.download('wordnet')"`
4) `./filter.py -f example.txt -n 10`

## Docker

1) Download the [model](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip) file, unzip it, put it inside the `/model` directory
2) `docker build . -t filter`
3) `docker run -i --rm filter -f - -n 10 < example.txt`

## Examples

Get the 10 "best" verbs from the list:

```
./filter.py -f example.txt -n 10 -p VERB

Average
Betray
Calm
Control
Create
Depart
Empty
Fancy
Release
Travel
```

Give me the "best" 100 words from all my wordlists:

```
find /path/to/wordlists -type f -name '*.txt' -exec cat {} \; | ./filter.py -f - -n 100

Advance
Advantage
Affect
Animal
Armed
...
```

# Why did you make this?

I wanted a way to take all the RPG Oracle wordlists I have, combine them together, and generate a single list that encompasses the best of all of them. More importantly, I wanted something that:

- Was deterministic. The same inputs will yield the same outputs.
- Could optionally filter words by their part-of-speech (noun, verb, adjective)
- Took word similarity and diversity into account
- Allowed me to specify priorities of words

# How do I prioritize words?

Two ways:

- Words that appear more than once in the list have higher priority
- Words that appear towards the top of the list have higher priority

This means you can concatenate a bunch of wordlists together in a preferred order, then run it through this filter to spit out the best words from those lists.

# How does it work?

Ah, thats the complicated part.

First, it cleans the input words, removing lines that start with comments, lines that contain multiple words, lines that contain symbols, etc.

Then it reorders the words by frequency, while maintaining the input order for words with the same counts.

Next, it optionally filters on the specified part of speech, leaving only nouns/verbs/adjectives in the list.

Then, it uses the `hdbscan` algorithm to cluster words together based on their vector representation.

> Word2Vec is a popular algorithm for word embedding, which represents words as dense vectors in a high-dimensional space based on their contextual usage patterns. It captures semantic and syntactic relationships between words, enabling mathematical operations on the word vectors, such as word similarity and analogy.

(This code uses the FastText model for converting words to vectors, because it can handle words outside of the training set)

`hdbscan` will group similar words together, such as `[dog, canine, puppy]`, into clusters.

The "best" word is chosen from each cluster, with the intent that the wordlist won't contain duplicate words in terms of meaning.

Then, from the remaining list, the `n` most diverse words are chosen. This is done by finding the words with the most _dissimilar_ word vectors.

The final list is sorted alphabetically. 

TLDR:
- Prioritize the words
- Cluster them by similarity
- Pick one word from each cluster to make a list
- Choose the N most dissimilar words from that list

# What to you mean by "best" words?

"Best" is subjective, but here is what the algorithm does. Given a cluster, it goes down this list of criteria:

- Take the cluster and all of its synonyms, calculate an average vector, and find the single word closest to that average vector. Skip this if the cluster is too small.
- Choose the word in the cluster that is the most frequent (due to duplicates). Skip if there is a tie.
- Choose the word that has the most synonyms and antonyms, indicating that it is a versatile word that could have multiple interpretations

# Where did the `example.txt` come from?

It is built from multiple wordlists, the primary ones coming from:

- [Mythic GME 2e by Tana Pigeon](https://wordmillgames.com/)
- [Ironsworn by Shawn Tomkin](https://www.ironswornrpg.com/)

# Why is your code so bad?

I pieced it together with the help of ChatGPT. It may not be the most efficient, but it is pretty well commented and broken out into small functions, so hopefully its easy to read/understand.
