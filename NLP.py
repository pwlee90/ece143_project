"""
This file is where we conducted our NLP. For our NLP, we load in the combined data file and
extract new features from the lyrics.
Specifically, we extract the following features:
"""

import pandas as pd
import re
from collections import defaultdict

def analyze_emotions(lyrics, lexicon):
    """
    Analyze the emotions present in the given lyrics using the provided lexicon.

    Args:
        lyrics (str): The lyrics text to analyze.
        lexicon (dict): A dictionary where keys are words and values are lists of emotions.

    Returns:
        defaultdict: A dictionary with emotions as keys and their counts as values.
    """
    emotions_count = defaultdict(int)
    # Tokenize the text: split into words
    words = re.findall(r'\b\w+\b', lyrics.lower())
    for word in words:
        if word in lexicon:
            for emotion in lexicon[word]:
                emotions_count[emotion] += 1
    return emotions_count

def normalize_emotions(lyrics, emotion_counts):
    """
    Normalize the emotion counts by the total number of words in the lyrics.

    Args:
        lyrics (str): The lyrics text.
        emotion_counts (dict): A dictionary with emotions as keys and their counts as values.

    Returns:
        dict: A dictionary with emotions as keys and their normalized scores as values.
    """
    total_words = len(re.findall(r'\b\w+\b', lyrics.lower()))
    # Avoid division by zero for empty lyrics.
    if total_words == 0:
        return {emotion: 0 for emotion in emotions_list}
    return {emotion: emotion_counts.get(emotion, 0) / total_words for emotion in emotions_list}

if __name__ == "__main__":
    # Load the existing CSV dataset.
    df = pd.read_csv("cleaned_data.csv")
    print("Dataset head:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nDataset description:")
    print(df.describe())

    # Load the NRC Emotion Lexicon.
    nrc_file = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    nrc_df = pd.read_csv(nrc_file, sep="\t", names=["word", "emotion", "value"])

    nrc_df = nrc_df[nrc_df["value"] == 1].drop(columns=["value"])

    nrc_dict = nrc_df.groupby("word")["emotion"].apply(list).to_dict()
    print("\nSample of NRC Lexicon:")
    print(nrc_df.head())


    # Define the list of emotions for which we want to add columns.
    emotions_list = ['negative', 'disgust', 'fear', 'sadness', 'positive', 
                    'anger', 'trust', 'anticipation', 'joy', 'surprise']

    # Apply the emotion analysis function to each row's lyrics.
    emotion_results = df['lyrics'].apply(lambda x: analyze_emotions(x, nrc_dict))

    # Add new columns to the DataFrame for each specified emotion.
    for emotion in emotions_list:
        df[emotion] = emotion_results.apply(lambda emo_dict: emo_dict.get(emotion, 0))

    print("\nDataset with added emotion score columns:")
    print(df.head())

    # Normalized scores (counts divided by total words in the lyrics)
    normalized_emotions = df['lyrics'].apply(lambda x: normalize_emotions(x, analyze_emotions(x, nrc_dict)))
    normalized_emotions_df = pd.DataFrame(normalized_emotions.tolist())

    # Rename normalized columns with a prefix, e.g., norm_anger, norm_joy, etc.
    normalized_emotions_df = normalized_emotions_df.add_prefix("norm_")

    # Merge the normalized emotion scores with the original DataFrame.
    df_normalized = pd.concat([df, normalized_emotions_df], axis=1)
    print("\nDataset with normalized emotion scores:")
    print(df_normalized.head())

    # Save the updated DataFrame to a new CSV file.
    df_normalized.to_csv("cleaned_data_with_emotions.csv", index=False)
