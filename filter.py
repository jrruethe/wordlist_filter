#!/usr/bin/env python3

import re
import os

from collections import Counter
from itertools import combinations

import nltk
import numpy as np
import hdbscan
import sklearn

from nltk.corpus import wordnet
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
from openai import OpenAI

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])

# Clean an input list of words
def clean_input(words):

  # Regular expressions
  article  = re.compile(r'^(a |an |the )?(.+)$', re.IGNORECASE)
  symbol   = re.compile(r'[ _-]', re.IGNORECASE)
  or_split = re.compile(r'^\w+ or \w+$', re.IGNORECASE) # TODO

  # Split words separated by "/"
  # such as "Driver/Pilot"
  flattened_words = []
  for word in words:
    # Strip whitespace
    word = word.strip()
    if not word:
      continue
    if word.startswith("#"):
      continue
    elif len(word) < 3:
      continue
    elif "/" in word:
      flattened_words += word.split("/")
    elif re.match(or_split, word):
      flattened_words += word.split(" or ")
    else:
      flattened_words.append(word)

  # Now loop through this list, which may be longer than the input
  cleaned = []
  for word in flattened_words:

    # Make the word lowercase
    word = word.lower()

    # If the word starts with an article, remove it
    match = article.match(word)
    if article.match(word):
      word = match.group(2)

    # If the word contains a symbol, ignore it
    if not symbol.search(word):
      cleaned.append(word)

  # Return the cleaned input
  return cleaned

# Returns the unique words without sorting
def uniq(words):
  return dict.fromkeys(words).keys()

# Take an input list, order them by frequency,
# then remove duplicates.
# Optionally remove words that are less frequent than N
def order_words_by_frequency(words, n=1):

  # Count the frequency of each word in the input list
  word_counts = Counter(words)

  # Sort the dictionary by value (frequency) and get a list of words
  ordered_words = sorted(word_counts, key=word_counts.get, reverse=True)

  # Remove duplicates from the list of words
  unique_words = uniq(ordered_words)

  # Filter out words with a count lower than n
  filtered_words = [word for word in unique_words if word_counts[word] >= n]

  return filtered_words

# Given a wordlist and a part of speech (VERB, NOUN, ADJECTIVE)
# return only the words from that list that match the POS.
# Some words have multiple POS based on context, this will return
# the word if it matches any of them.
def part_of_speech(wordlist, pos):

  # Map parts of speech to WordNet constants
  mapping = {
    "VERB"     : wordnet.VERB,
    "ADJECTIVE": wordnet.ADJ,
    "NOUN"     : wordnet.NOUN,
    "ADVERB"   : wordnet.ADV,
  }
  wordnet_pos = mapping[pos]

  # Filter the words
  filtered = []
  for word in wordlist:
    synsets = wordnet.synsets(word)
    for synset in synsets:
      if word in synset.name():
        if synset.pos() == wordnet_pos:
          filtered.append(word)
          break

  return filtered

# Given a word list,
# returns a list of words paired with their vectors using the fasttext model
# words = [word]
# Returns [(word, vector)]
def get_word_vectors(words):
  word_vectors = []
  for word in words:

    # Get the embedding
    response = client.embeddings.create(input=word, model=os.environ["OPENAI_MODEL"])
    embedding = response.data[0].embedding

    word_vectors.append((word, embedding))
  return word_vectors


# Given a list of wordvector pairs [(word, vector)],
# returns a list of clusters of word vector pairs, and a list of remainders
# Given a list of words, returns a list of clusters and remainders
# words = [word]
# Returns [[word]], [word]
def cluster_words(words):

  # Get the word vectors
  word_vectors = get_word_vectors(words)

  # Cluster the words
  # https://github.com/scikit-learn-contrib/hdbscan/issues/69
  clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1)
  vectors = np.array([v[1] for v in word_vectors])
  distance = sklearn.metrics.pairwise_distances(vectors, metric="cosine")
  cluster_labels = clusterer.fit_predict(distance.astype("float64"))

  # Pull out the remainders
  remainders = [word_vectors[i] for i in np.where(cluster_labels == -1)[0]]
  remainders = [pair[0] for pair in remainders]

  # Pull out the clusters
  clusters = []
  for label in set(cluster_labels):
    if label == -1:
      continue
    clusters.append([word_vectors[i] for i in np.where(cluster_labels == label)[0]])

  # Pull the words out of the clusters
  clusters = [[pair[0] for pair in cluster] for cluster in clusters]

  return clusters, remainders

# Given a word, return all the synonyms for the word
def synonyms(word, pos=None):
  synonyms = []

  # Map parts of speech to WordNet constants
  mapping = {
    "VERB"     : wordnet.VERB,
    "ADJECTIVE": wordnet.ADJ,
    "NOUN"     : wordnet.NOUN,
    "ADVERB"   : wordnet.ADV,
    None : None,
  }
  wordnet_pos = mapping[pos]

  for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
      synonym = lemma.name()
      if wordnet_pos is None or lemma.synset().pos() == wordnet_pos:
        synonyms.append(synonym)

  return synonyms

# Given a word, return all the antonyms for the word
def antonyms(word, pos=None):
  antonyms = []

  # Map parts of speech to WordNet constants
  mapping = {
    "VERB"     : wordnet.VERB,
    "ADJECTIVE": wordnet.ADJ,
    "NOUN"     : wordnet.NOUN,
    "ADVERB"   : wordnet.ADV,
    None : None,
  }
  wordnet_pos = mapping[pos]

  for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
      if lemma.antonyms():
        for antonym in lemma.antonyms():
          if wordnet_pos is None or antonym.synset().pos() == wordnet_pos:
            antonyms.append(antonym.name())

  return antonyms

# Takes in two wordlists
# Returns the set difference, maintaining order
# a,b = [word]
# Returns [word]
def set_difference(a, b):
  b_set = set(b)
  c = [word for word in a if word not in b_set]
  return c

# Takes in two wordlists
# Returns the set intersection, maintaining order
# a,b = [word]
# Returns [word]
def set_intersection(a, b):
  b_set = set(b)
  c = [word for word in a if word in b_set]
  return c

# Takes in two wordlists
# Returns the set union, maintaining order without introducing duplicates
# a,b = [word]
# Returns [word]
def set_union(a, b):
  b_set = set(b)
  c = [word for word in a]
  for word in b:
    if word not in b_set:
      c.append(word)
      b_set.add(word)
  return c

# Given a wordlist, return a wordlist
# that also contains all the synonyms of each word
def augment(wordlist, pos=None):

  # Augment the wordlist with the synonyms of each word
  augmentation = []
  for word in wordlist:
    augmentation += synonyms(word, pos=pos)

  # Remove words from the augmentation that already appear in the wordlist
  words_to_append = set_difference(augmentation, wordlist)

  # Augment the wordlist
  return wordlist + augmentation


# Given a cluster, pick the word closest to the average vector of that cluster
# Automatically gravitates towards duplicates
# cluster = [word]
# Returns "word" or None
# Can pass in a second cluster that is used to calculate the mean
# mean_cluster should be >= cluster
def select_average_word(cluster, mean_cluster=None):
  if mean_cluster == None:
    mean_cluster = cluster

  # Doesn't work for small clusters
  if len(mean_cluster) <= 2:
    return None

  # Calculate the mean vector for the cluster
  mean_word_vectors = get_word_vectors(mean_cluster)
  mean_vector = np.mean([pair[1] for pair in mean_word_vectors], axis=0)

  # Find the word with the closest vector to the mean vector
  word_vectors = get_word_vectors(cluster)
  closest_pair = word_vectors[0]
  closest_distance = cosine(mean_vector, word_vectors[0][1])
  for pair in word_vectors[1:]:
    distance = cosine(mean_vector, pair[1])
    if distance < closest_distance:
      closest_distance = distance
      closest_pair = pair

  return closest_pair[0]

# Given a cluster, pick the word from the cluster that is the most frequent
# Returns None if there is a tie
# cluster = [word]
# Returns "word" or None
def select_most_frequent_word(cluster):
  # Count the words
  word_count = Counter(cluster)

  # Find the word(s) with the highest count
  max_count = max(word_count.values())
  frequent_words = [word for word, count in word_count.items() if count == max_count]

  # Check if there is more than one word with the highest count
  if len(frequent_words) > 1:
    return None
  else:
    return frequent_words[0]

# Pass in an ordered wordlist as a ranking
# Given a cluster, pick the word from the cluster that has the highest ranking
# cluster = [word]
# Returns "word" or None
def select_highest_ranked_word(cluster, ranking):
  for rank in ranking:
    for word in cluster:
      if rank == word:
        return word
  return None

# Given a cluster, pick the word that has the most number of synonyms+antonyms
# On a tie, returns the first one found
# cluster = [word]
# Returns "word" or None
def select_most_versatile_word(cluster, pos=None):
    choice = None
    max_value = float('-inf')
    for word in cluster:
      value = len(synonyms(word, pos=pos)) + len(antonyms(word, pos=pos))
      if value > max_value:
        max_value = value
        choice = word
    return choice

# Uses a variety of methods to select the best word from a cluster
# cluster = [word]
# Returns "word" or None
def select_best_word(cluster, ranking=None, pos=None):
  best_word = None

  # Try to choose the highest ranked word
  if ranking:
    best_word = select_highest_ranked_word(cluster, ranking)

  # If there is no ranking, choose the average
  if not best_word:

    # Augment the cluster that is used to generate the mean vector,
    # but then use the original cluster to select the word closest to the mean vector
    augmented_cluster = augment(cluster, pos=pos)
    best_word = select_average_word(cluster, augmented_cluster)

  # If we are unable to do an average, choose the most frequent word
  if not best_word:
    best_word = select_most_frequent_word(cluster)

  # If there isn't a most frequent word, choose the most versatile word
  if not best_word:
    best_word = select_most_versatile_word(cluster, pos=pos)

  return best_word

# clusters = [[word]]
# Returns [word]
def select_best_words(clusters, ranking=None, pos=None):
  output = []
  for cluster in clusters:
    output.append(select_best_word(cluster, ranking, pos=pos))
  return output


# Given a wordlist, select the N most diverse words
# calculated by picking word vectors that are furthest
# away from each other
def max_diversity_sampling(words, n):

  # Not enough words
  if n >= len(words):
    return words

  # Calculate the word vectors
  word_vectors = get_word_vectors(words)

  # Calculate the cosine distances of the word vectors
  vectors = np.array([v[1] for v in word_vectors])
  distance = pairwise_distances(vectors, metric="cosine")

  # Start with the first word in the list
  selected_indices = [0]

    # Select the remaining n-1 words that are the most diverse
  for i in range(n-1):

    # Calculate the minimum distance between the selected words and all other words
    min_distances = np.min(distance[selected_indices], axis=0)

    # Select the word with the maximum minimum distance from the selected words
    max_distance_index = np.argmax(min_distances)
    selected_indices.append(max_distance_index)

  return [word_vectors[i][0] for i in selected_indices]

# Capitalize each word and remove duplicates
def clean_output(words):
  output = []
  for word in words:
    output.append(word.capitalize())
  return uniq(output)


if __name__ == '__main__':
  import argparse
  import sys

  # Set up command line argument parser
  parser = argparse.ArgumentParser(description='Given a list of words, return the best subset of words')
  parser.add_argument('-f', '--file', type=str, help='File containing a list of words. Use - for STDIN')
  parser.add_argument('-p', '--pos', type=str, help='One of VERB,ADJECTIVE,NOUN')
  parser.add_argument('-n', '--num', type=int, help='Number of words to select')

  # Parse command line arguments
  args = parser.parse_args()

  if not args.file:
    parser.print_help()
    sys.exit(1)

  # Read the wordlist
  if args.file == "-":
    word_list = [word.strip() for word in sys.stdin.readlines()]
  else:
    with open(args.file, 'r') as file:
      word_list = [word.strip() for word in file.readlines()]

  # Clean the input
  word_list = clean_input(word_list)

  # Get the words ordered by frequency
  word_list = order_words_by_frequency(word_list)

  # Filter the wordlist down to the part of speech
  if args.pos:
    word_list = part_of_speech(word_list, pos=args.pos)

  # Cluster the words
  clusters, remainders = cluster_words(word_list)

  # Select the best words from each cluster
  best_words = select_best_words(clusters, pos=args.pos)

  # Reorder the words to match the input
  best_words = set_intersection(word_list, best_words + remainders)

  # Select the N most diverse words
  if args.num:

    # I have found that this tends to give better results
    # This restricts the diversity selection to the most frequent words
    best_words = best_words[0:args.num*2]
    best_words = max_diversity_sampling(best_words, args.num)

  # Clean the words
  output = clean_output(best_words)

  # Output the words
  for word in sorted(output):
    print(word)
