from bs4 import BeautifulSoup
from random import randrange

true_list = list()
false_list = list()
max_leng = 0
count = 0
all_pos = list()
entity_type = ['Individual_protein', 'Protein_family_or_group', 'Gene/protein/RNA', 'Gene', 'DNA_family_or_group', 'Protein_complex']


def mark_entity(sentence_text, interact, pos_list, **entity_dict):  # entity_dict = {eid:[entity, charOffset, type]}
    global max_leng, count
    for item in interact:    # interact [e1, e2, interaction]
        entity1 = find_offset(item[0], **entity_dict)
        entity2 = find_offset(item[1], **entity_dict)
        e1_offset = entity1[1]   # 实体位置
        e2_offset = entity2[1]
        e1_type = entity1[2]
        e2_type = entity2[2]

        # e1_offset = e1_offset.split('-')
        # e2_offset = e2_offset.split('-')

        if ',' in e1_offset:
            e1_offset = e1_offset.split(',')[1].split('-')
        else:
            e1_offset = e1_offset.split('-')
        if ',' in e2_offset:
            e2_offset = e2_offset.split(',')[1].split('-')
        else:
            e2_offset = e2_offset.split('-')

        sent = list(sentence_text)
        e1 = "".join(sent[int(e1_offset[0]):int(e1_offset[1]) + 1])
        e2 = "".join(sent[int(e2_offset[0]):int(e2_offset[1]) + 1])

        if int(e1_offset[0]) > int(e2_offset[0]):
            sent.insert(int(e2_offset[0]), ' <*> ')
            sent.insert(int(e2_offset[1]) + 1 + 1, ' </*> ')
            sent.insert(int(e1_offset[0]) + 1 + 1, ' <#> ')
            sent.insert(int(e1_offset[1]) + 1 + 3, ' </#> ')
        else:
            sent.insert(int(e1_offset[0]), ' <*> ')
            sent.insert(int(e1_offset[1]) + 2, ' </*> ')
            sent.insert(int(e2_offset[0]) + 2, ' <#> ')
            sent.insert(int(e2_offset[1]) + 4, ' </#> ')

        sent = ''.join(sent)

        if int(e1_offset[0]) > int(e2_offset[0]):
            t = e2
            e2 = e1
            e1 = t

        if item[2] == 'True':
            true_list.append([e1, e1_type, e2, e2_type, sent])
        else:
            false_list.append([e1, e1_type, e2, e2_type, sent])


def find_offset(eid, **entity_dict):
    for k, v in entity_dict.items():
        if eid == k:
            return v


def parse_xml(soup):      # 解析xml
    global max_sent_len
    documents = soup.find_all('document')
    for document in documents:
        sentence = document.find('sentence')
        while sentence:
            # sentence
            sentence_text = sentence['text']
            # entity
            entity_dict = dict()   # eid:[entity, charOffset]
            all_entity = list()  # 每个sentence中所有的实体
            entitys = sentence.find_all('entity')
            for e in entitys:
                entity_dict[e['id']] = [e['text'], e['charOffset'], e['type']]
            # pos
            pos_list = list()
            tokenize = sentence.find('tokenization', {"tokenizer": "split"})
            tokens = tokenize.find_all('token')
            for token in tokens:
                pos_list.append([token['text'], token['charOffset'], token['POS']])
                if token['POS'] not in all_pos:
                    all_pos.append(token['POS'])
            # pair
            pairs = sentence.find_all('pair')
            interact = list()  # [ [e1, e2, interaction], [e1, e2, interaction],...]
            if pairs:
                for p in pairs:
                    interact.append([p['e1'], p['e2'], p['interaction']])
                mark_entity(sentence_text, interact, pos_list, **entity_dict)
            sentence = sentence.next_sibling.next_sibling  # 下一个节点，过滤换行符，因为这里把换行符当作了一个节点
    return


def split_data():    # 多通道格式 8:2
    train = list()
    for i in range(int(len(true_list) * 0.8)):  # train 正例
        num = randrange(0, len(true_list))
        train.append([true_list[num], 'True'])
        del true_list[num]
    for i in range(0, int(len(false_list) * 0.8)):    # train 负例
        num = randrange(0, len(false_list))
        train.append([false_list[num], 'False'])
        del false_list[num]
    count = 0
    # write train
    for i in range(0, len(train)):   # 打乱再写入
        count += 1
        num = randrange(0, len(train))
        write_file("train.txt", count, train[num][0][2], train[num][1])
        del train[num]
    print(count)
    # write test
    test_data = list()
    for i in range(0, len(true_list)):
        test_data.append([true_list[i], 'True'])
    for i in range(0, len(false_list)):
        test_data.append([false_list[i], 'False'])
    test_len = len(test_data)
    for i in range(test_len):
        count += 1
        num = randrange(0, len(test_data))
        write_file("test.txt", count, test_data[num][0][2], test_data[num][1])
        del test_data[num]
    print(count)
    return


def write_file(filename, count, text, interact):
    with open(filename, "a", encoding='utf-8') as file:
        #file.write(str(count) + '\t' + text + '\n' + interact + '\n\n')
        file.write(text + '\n')
    return


def split_data2():  # bert格式6:2:2
    true6 = int(len(true_list) * 0.6)
    false6 = int(len(false_list) * 0.6)
    true2 = int(len(true_list) * 0.2)
    false2 = int(len(false_list) * 0.2)
    # train
    train = list()
    for i in range(true6):  # train 正例
        num = randrange(0, len(true_list))
        train.append([true_list[num], 'True'])
        del true_list[num]
    for i in range(false6):  # train 负例
        num = randrange(0, len(false_list))
        train.append([false_list[num], 'False'])
        del false_list[num]
    # dev
    dev = list()
    for i in range(true2):  # dev 正例
        num = randrange(0, len(true_list))
        dev.append([true_list[num], 'True'])
        del true_list[num]
    for i in range(false2):  # dev 负例
        num = randrange(0, len(false_list))
        dev.append([false_list[num], 'False'])
        del false_list[num]
    count = 0
    train_len = len(train)
    dev_len = len(dev)
    # write train
    for i in range(train_len):
        count += 1
        num = randrange(0, len(train))
        with open("train.txt", "a", encoding='utf-8') as file:
            # e1 \t e1_type \t e2 \t e2_type \t sentence \t interact
            file.write(train[num][0][0] + '\t' + train[num][0][1] + '\t' + train[num][0][2] + '\t' + train[num][0][3] + '\t' + train[num][0][4] + '\t' + train[num][1] + '\n')
            # e1 \t e2 \t sentence \t interact
            # file.write(train[num][0][0] + '\t' + train[num][0][1] + '\t' + train[num][0][2] + '\t' + train[num][1] + '\n')
        del train[num]
    print(count)
    # write dev
    for i in range(dev_len):
        count += 1
        num = randrange(0, len(dev))
        with open("dev.txt", "a", encoding='utf-8') as file:
            file.write(dev[num][0][0] + '\t' + dev[num][0][1] + '\t' + dev[num][0][2] + '\t' + dev[num][0][3] + '\t' + dev[num][0][4] + '\t' + dev[num][1] + '\n')
            # e1 \t e2 \t sentence \t interact
            # file.write(dev[num][0][0] + '\t' + dev[num][0][1] + '\t' + dev[num][0][2] + '\t' + dev[num][1] + '\n')
        del dev[num]
    print(count)
    # write test
    test_data = list()
    for i in range(0, len(true_list)):
        test_data.append([true_list[i], 'True'])
    for i in range(0, len(false_list)):
        test_data.append([false_list[i], 'False'])
    test_len = len(test_data)
    for i in range(test_len):
        count += 1
        num = randrange(0, len(test_data))
        with open("test.txt", "a", encoding='utf-8') as file:
            file.write(test_data[num][0][0] + '\t' + test_data[num][0][1] + '\t' + test_data[num][0][2] + '\t' + test_data[num][0][3] + '\t' + test_data[num][0][4] + '\t' + test_data[num][1] + '\n')
            # file.write(test_data[num][0][0] + '\t' + test_data[num][0][1] + '\t' + test_data[num][0][2] + '\t' + test_data[num][1] + '\n')
        del test_data[num]
    print(count)
    return


def split_data3():   # bert格式8:2
    while len(false_list) > 3*len(true_list):
        num = randrange(0, len(false_list))
        del false_list[num]
    print(len(true_list), len(false_list))
    true6 = int(len(true_list) * 0.8)
    false6 = int(len(false_list) * 0.8)
    # train
    train = list()
    for i in range(true6):  # train 正例
        num = randrange(0, len(true_list))
        train.append([true_list[num], 'True'])
        del true_list[num]
    for i in range(false6):  # train 负例
        num = randrange(0, len(false_list))
        train.append([false_list[num], 'False'])
        del false_list[num]

    count = 0
    train_len = len(train)
    # write train
    for i in range(train_len):
        count += 1
        num = randrange(0, len(train))
        with open("train.txt", "a", encoding='utf-8') as file:  # e1 \t e2 \t sentence \t interact
            file.write(train[num][0][0] + '\t' + train[num][0][1] + '\t' + train[num][0][2] + '\t' + train[num][1] + '\n')
        del train[num]
    print(count)

    # write test
    test_data = list()
    for i in range(0, len(true_list)):
        test_data.append([true_list[i], 'True'])
    for i in range(0, len(false_list)):
        test_data.append([false_list[i], 'False'])
    test_len = len(test_data)
    for i in range(test_len):
        count += 1
        num = randrange(0, len(test_data))
        with open("test.txt", "a", encoding='utf-8') as file:  # e1 \t e2 \t sentence \t interact
            file.write(test_data[num][0][0] + '\t' + test_data[num][0][1] + '\t' + test_data[num][0][2] + '\t' +
                       test_data[num][1] + '\n')
        del test_data[num]
    print(count)


with open("./bioCorpus/BioInfer.xml", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, 'xml')
    parse_xml(soup)
    print(len(all_pos), all_pos)
    print(len(entity_type), entity_type)
    split_data()
