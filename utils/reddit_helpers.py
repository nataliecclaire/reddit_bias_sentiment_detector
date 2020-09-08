import requests
import json
import re


def get_pushshift_submissions_interval(query, sub, after, before):
    url = 'https://api.pushshift.io/reddit/search/submission/?q='+str(query)+'&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def get_pushshift_comments(q1, q2, s, b, a):
    url = 'https://api.pushshift.io/reddit/search/comment/?q='+str(q1)+'+'+str(q2)+'&size='+str(s)+'&after='+str(a)+'&before='+str(b)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def get_pushshift_submissions(query, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?q='+str(query)+'&size=1000'+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def process_reddit(comment):
    comment = comment.encode("ascii", errors="ignore").decode()
    comment = re.sub('[^A-Za-z,. ]+', '', comment)
    return comment


def process_tweet(sent):
    # special cases - 15959
    # print(sent)
    sent = sent.encode("ascii", errors="ignore").decode() # check this output
    # print(sent)
    sent = re.sub('@[^\s]+', '', sent)
    sent = re.sub('https: / /t.co /[^\s]+', '', sent)
    sent = re.sub('http: / /t.co /[^\s]+', '', sent)
    sent = re.sub('http[^\s]+', '', sent)

    sent = re.sub('&gt', '', sent)

    # split camel case combined words
    sent = re.sub('([A-Z][a-z]+)', r'\1', re.sub('([A-Z]+)', r' \1', sent))

    sent = sent.lower()

    # remove numbers
    sent = re.sub(' \d+', '', sent)
    # remove words with letter+number
    sent = re.sub('\w+\d+|\d+\w+', '', sent)

    # remove spaces
    sent = re.sub('[\s]+', ' ', sent)
    sent = re.sub('[^\w\s.!\-?]', '', sent)

    # remove 2 or more repeated char
    sent = re.sub(r"(.)\1{2,}", r"\1", sent)
    sent = re.sub(" rt ", "", sent)

    sent = re.sub('- ', '', sent)

    sent = sent.strip()

    # print(sent)
    return sent