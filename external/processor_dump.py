import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    with open('corpus.txt', mode='rt', encoding='utf-8') as f:
        read_data = f.readlines()

    titles = []
    texts = []
    is_title = True
    body = []
    read_data.append('\n')
    for line in tqdm(read_data):
        if line == '\n':
            is_title = True
            texts.append(''.join(body)[:-2])
            body = []
        elif is_title:
            titles.append(line.replace('\n', ''))
            is_title = False
        else:
            body.append(line)
    df = pd.DataFrame({
        'title': titles,
        'excerpt': texts
    })
    df.to_csv('dump_of_simple_english_wiki.csv', index=False)
    df.head()
