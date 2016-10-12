import re
import json
import html
import random
from urllib.request import urlopen


link = re.compile(r"<a.*?>.*?</a>")
html_tag = re.compile(r"<.+?>")
newline = re.compile(r"<br.*?>")

common_regex = [
    re.compile(r'\d+'),
    re.compile(r'\b\w{1,3}\b')
]
common = [
    'что', 'это', 'то', 'ты', 'по', 'за', 'так', 'же', 'но', 'все', 'из', 'от', 'там', 'если', 'ну', 'есть',
    'да', 'бы', 'вот', 'только', 'даже', 'уже', 'для', 'можно', 'еще', 'было', 'или', 'ну', 'он', 'ли',
    'ты', 'они', 'до', 'мне', 'да', 'тебя'
]

def get_threads(board):
    board = "https://2ch.hk/{board}/catalog.json".format(board=board)
    data = json.loads(urlopen(board).read().decode('utf8'))
    return list(map(lambda x: x["num"], data["threads"]))


def get_thread(board, thread):
    thread = "https://2ch.hk/{board}/res/{thread}.json".format(board=board, thread=thread)

    try:
        data = json.loads(urlopen(thread).read().decode('utf8'))
        text = "\n".join(list(map(lambda x: x["comment"], data["threads"][0]["posts"])))
        text = re.sub(link, '', text)
        text = re.sub(newline, '. ', text)
        text = re.sub(html_tag, '', text)

        text = html.unescape(text)
        text = text.replace('>', '')

        print(thread + " done")
        return text
    except UnicodeDecodeError as e:
        print(e)
        return None


def collect(boards, threads_per_board, random_state=None):
    d = {}
    if random_state:
        random.seed(random_state)
    for e in boards:
        try:
            with open('boards/{}.json'.format(e), 'r', encoding='utf8') as f:
                t = json.load(f)
        except FileNotFoundError:
            threads = get_threads(e)
            t = []
            count = min(threads_per_board, len(threads))
            thread_names = random.sample(threads, count)
            for thread in thread_names:
                data = get_thread(e, thread)
                if data:
                    t.append(data)

            with open('boards/{}.json'.format(e), 'w+', encoding='utf8') as f:
                json.dump(t, f, ensure_ascii=False)
        d[e] = t
    return d


def get_random_thread(board):
    return get_thread(board, random.choice(get_threads(board)))


def remove_common_words(text):
    t = text
    for e in common:
        t = t.replace(e, '')
    for e in common_regex:
        t = e.sub('', t)
    return t
