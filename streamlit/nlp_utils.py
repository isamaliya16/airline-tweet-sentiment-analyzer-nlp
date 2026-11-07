import re

negation_words = ['arent', 'cannot', 'cant', 'couldnt', 'didnt', 'doesnt', 'dont', 'isnt', 'never', 'no', 'nor', 'not', 'shouldnt', 'wasnt', 'wont', 'wouldnt']
positive_words = ['amazing', 'appreciate', 'appreciated', 'awesome', 'best', 'comfortable', 'early', 'easy', 'enjoy', 'enjoyed', 'excellent', 'fantastic', 'fast', 'friendly', 'good', 'great', 'happy', 'helpful', 'impressed', 'kudos', 'love', 'loved', 'nice', 'outstanding', 'perfect', 'pleasant', 'polite', 'professional', 'quick', 'resolved', 'safe', 'smooth', 'thank', 'thanks', 'welcome', 'wonderful']
negative_words = ['angry', 'awful', 'bad', 'broken', 'canceled', 'cancellation', 'cancelled', 'complaint', 'damaged', 'delay', 'delayed', 'delays', 'disappointed', 'disappointing', 'disgusting', 'failed', 'hate', 'hold', 'horrible', 'late', 'lost', 'missed', 'nightmare', 'overbooked', 'poor', 'refund', 'ridiculous', 'rude', 'slow', 'stuck', 'terrible', 'unacceptable', 'unhelpful', 'useless', 'waiting', 'worse', 'worst']


def clean_text(s):
    s = s.lower()
    s = re.sub(r'https?://\S+', ' ', s)      # remove URLs
    s = re.sub(r'@\w+', ' ', s)              # remove @mentions
    s = re.sub(r'#(\w+)', r'\1', s)          # keep hashtag word
    s = re.sub(r"['’‘]", '', s)             # don't -> dont
    s = re.sub(r'[^a-z\s]', ' ', s)          # numbers, punctuation, emojis
    s = re.sub(r'\s+', ' ', s).strip()       # extra spaces
    return s


def heuristic_score(text):
    words = clean_text(text).split()
    score = 0
    for i, w in enumerate(words):
        if w in positive_words:
            value = 1
        elif w in negative_words:
            value = -1
        else:
            continue
        if any(p in negation_words for p in words[max(0, i - 2):i]):
            value = -value
        score += value
    return score


def heuristic_sentiment(text):
    score = heuristic_score(text)
    if score > 0:
        return 'positive'
    if score < 0:
        return 'negative'
    return 'neutral'
