try:
    from nltk.corpus import wordnet
except:
    print ("download nltk stuff for word sense disambiguation")
    import nltk
    nltk.download('wordnet')
    nltk.download('stops')
    from nltk.corpus import wordnet


from nym_embeddings.pywsd.allwords_wsd import disambiguate_tokens

# loading all wordnet once, to get all synsets loaded
alle = list(wordnet.all_synsets())

def lazy_lemmatize_tokens(tokens):
    return disambiguate_tokens(tokens)


