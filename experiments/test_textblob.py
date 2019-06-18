from textblob import TextBlob


if __name__ == '__main__':
    word_sent = TextBlob("I really don't know what to think of your actions today. I'm quite disappointed.").sentiment
    print(word_sent.polarity, word_sent.subjectivity)