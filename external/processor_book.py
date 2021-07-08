import pandas as pd
from tqdm import tqdm


filenames = [
    'data/cbtest_CN_test_2500ex.txt',
    'data/cbtest_P_valid_2000ex.txt',
    'data/cbtest_CN_train.txt',
    'data/cbtest_V_test_2500ex.txt',
    'data/cbtest_CN_valid_2000ex.txt',
    'data/cbtest_V_train.txt',
    'data/cbtest_NE_test_2500ex.txt',
    'data/cbtest_V_valid_2000ex.txt',
    'data/cbtest_NE_train.txt',
    'data/cbt_test.txt',
    'data/cbtest_NE_valid_2000ex.txt',
    'data/cbt_train.txt',
    'data/cbtest_P_test_2500ex.txt',
    'data/cbt_valid.txt',
    'data/cbtest_P_train.txt'
]


if __name__ == '__main__':
    texts = []
    for filename in filenames:
        with open(filename, mode='rt', encoding='utf-8') as f:
            read_data = f.readlines()
        body = []
        read_data.append('\n')
        for line in tqdm(read_data):
            if line == '\n':
                texts.append(''.join(body)[:-2])
                body = []
            else:
                body.append(' '.join(line.split()[1:]))

    print(len(texts))
    df = pd.DataFrame({
        'excerpt': texts
    })
    df.to_csv('the_childrens_book_test.csv', index=False)
    df.head()
